#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification, Swinv2ForImageClassification

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import OpenFakeDataset  # noqa: E402
from data.collator import OpenFakeCollator  # noqa: E402
from training.metrics import compute_per_model_metrics  # noqa: E402

BASE_SWINV2 = "microsoft/swinv2-small-patch4-window16-256"
WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.pt",
    "checkpoint.pth",
    "checkpoint.pt",
    "weights.pth",
    "weights.pt",
)


def _base_swinv2_ref() -> str:
    for root in (os.environ.get("TRANSFORMERS_CACHE"), os.environ.get("HF_HOME")):
        if not root:
            continue
        cache_root = Path(root)
        candidates = [
            cache_root / "models--microsoft--swinv2-small-patch4-window16-256" / "snapshots",
            cache_root / "hub" / "models--microsoft--swinv2-small-patch4-window16-256" / "snapshots",
        ]
        for snapshots in candidates:
            if not snapshots.exists():
                continue
            for snapshot in sorted(snapshots.iterdir()):
                if (snapshot / "config.json").exists() and (snapshot / "preprocessor_config.json").exists():
                    return str(snapshot)
    return BASE_SWINV2


def samples_from_csv(test_csv: str):
    df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
    exists = df["file_name"].map(lambda p: Path(p).exists())
    missing = int((~exists).sum())
    if missing:
        print(f"[WARN] {test_csv}: skipping {missing:,}/{len(df):,} rows with missing image files")
        df = df.loc[exists].reset_index(drop=True)
    labels = df["label"].str.lower().eq("fake").astype(int)
    models = df["model"].astype(str) if "model" in df else pd.Series([""] * len(df))
    return list(zip(df["file_name"].astype(str), labels, models))


def _load_processor(ckpt: str):
    ckpt_path = Path(ckpt)
    base_ref = _base_swinv2_ref()
    if (ckpt_path.is_absolute() or ckpt_path.exists()) and (
        not ckpt_path.exists() or not (ckpt_path / "preprocessor_config.json").exists()
    ):
        ckpt = base_ref
    try:
        return AutoImageProcessor.from_pretrained(ckpt, backend="torchvision")
    except (OSError, TypeError):
        # Plain PyTorch checkpoints usually do not ship preprocessor_config.json.
        return AutoImageProcessor.from_pretrained(base_ref, backend="torchvision")


def _find_weight_file(ckpt: Path) -> Path:
    if ckpt.is_file():
        return ckpt
    for filename in WEIGHT_FILENAMES:
        candidate = ckpt / filename
        if candidate.exists():
            return candidate
    matches = sorted(
        p for pattern in ("*.safetensors", "*.bin", "*.pth", "*.pt")
        for p in ckpt.glob(pattern)
        if p.is_file()
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No model weights found under {ckpt}. Expected one of: {', '.join(WEIGHT_FILENAMES)}"
    )


def _extract_state_dict(state):
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict", "net"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")

    prefixes = ("module.", "model.")
    for prefix in prefixes:
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    return state


def _load_plain_swinv2_checkpoint(ckpt: Path, device: str) -> torch.nn.Module:
    weight_file = _find_weight_file(ckpt)
    config = AutoConfig.from_pretrained(_base_swinv2_ref())
    config.num_labels = 2
    config.id2label = {0: "real", 1: "fake"}
    config.label2id = {"real": 0, "fake": 1}
    model = Swinv2ForImageClassification(config)
    if weight_file.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weight_file))
    else:
        state = torch.load(str(weight_file), map_location="cpu")
    state = _extract_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 20 or len(unexpected) > 20:
        raise RuntimeError(
            f"{weight_file} does not look compatible with {BASE_SWINV2}: "
            f"{len(missing)} missing keys, {len(unexpected)} unexpected keys"
        )
    return model.to(device).eval()


def _load_model(ckpt: str, device: str) -> torch.nn.Module:
    ckpt_path = Path(ckpt)
    if ckpt_path.is_absolute() or ckpt_path.exists():
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {ckpt_path}. "
                "Verify the download/materialization step or update CHECKPOINTS in eval_existing_swinv2.py."
            )
        if ckpt_path.is_dir() and (ckpt_path / "config.json").exists():
            return AutoModelForImageClassification.from_pretrained(str(ckpt_path)).to(device).eval()
        return _load_plain_swinv2_checkpoint(ckpt_path, device)
    return AutoModelForImageClassification.from_pretrained(ckpt).to(device).eval()


def evaluate_checkpoint(
    ckpt: str,
    test_csv: str,
    out_csv: str,
    batch_size: int = 64,
    num_workers: int = 4,
    write_predictions: bool = False,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(ckpt, device)
    processor = _load_processor(ckpt)
    ds = OpenFakeDataset(samples_from_csv(test_csv), on_error="blank")
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=OpenFakeCollator(processor))
    logits, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=Path(test_csv).stem):
            labels.append(batch.pop("labels").cpu())
            batch = {k: v.to(device) for k, v in batch.items()}
            logits.append(model(**batch).logits.cpu())
    pred = torch.cat(logits).numpy()
    y = torch.cat(labels).numpy()
    probs = torch.softmax(torch.from_numpy(pred).float(), dim=-1).numpy()[:, 1]
    finite = np.isfinite(probs)
    roc_auc = None
    if np.any(finite) and len(np.unique(y[finite])) > 1:
        roc_auc = float(roc_auc_score(y[finite], probs[finite]))
    model_names = [s[2] for s in ds.samples]
    df = compute_per_model_metrics(pred, y, model_names)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if write_predictions:
        pd.DataFrame(
            {
                "file_name": [s[0] for s in ds.samples],
                "label": y,
                "model": model_names,
                "prob_fake": probs,
                "pred": (probs >= 0.5).astype(int),
            }
        ).to_csv(out.with_suffix(out.suffix + ".predictions.csv"), index=False)
    with open(str(out) + ".json", "w") as f:
        json.dump(
            {
                "checkpoint": ckpt,
                "test_csv": test_csv,
                "rows": len(ds),
                "roc_auc": roc_auc,
                "roc_AUC": roc_auc,
                "auc_roc": roc_auc,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--write_predictions", action="store_true")
    args = ap.parse_args()
    evaluate_checkpoint(
        args.checkpoint,
        args.test_csv,
        args.out_csv,
        args.batch_size,
        args.num_workers,
        write_predictions=args.write_predictions,
    )


if __name__ == "__main__":
    main()
