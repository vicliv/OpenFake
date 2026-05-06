#!/usr/bin/env python3
"""Community Forensics pretrained checkpoint evaluator.

Preprocessing (pinned from OwensLab/commfor-model-384 config.json + timm defaults):
  - Model : vit_small_patch16_384 (timm), loaded from OwensLab/commfor-model-384
  - Input size : 384 × 384
  - Resize mode : bicubic, then center-crop to 384
  - Normalization : mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)  [timm default for this ckpt]
  - Output : single logit; sigmoid > 0.5 → fake (label=1)

When test_csv has type='cf' rows, images are decoded from the HF dataset
(OwensLab/CommunityForensics-Eval) directly from the 'image_data' bytes field.
For all other CSVs the 'file_name' column is treated as an absolute filesystem path.
"""

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
import timm
from PIL import Image
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.metrics import compute_per_model_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# CF preprocessing (matches timm vit_small_patch16_384 defaults for this ckpt)
# ---------------------------------------------------------------------------
CF_TRANSFORM = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


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


def load_cf_model(device: str = "cuda") -> torch.nn.Module:
    """Load OwensLab/commfor-model-384 into a timm vit_small_patch16_384."""
    ckpt_path = hf_hub_download("OwensLab/commfor-model-384", "model.safetensors", repo_type="model")
    state = load_file(ckpt_path)
    # Strip the 'vit.' prefix that wraps each key
    state_bare = {k[len("vit."):]: v for k, v in state.items()}
    model = timm.create_model("vit_small_patch16_384", pretrained=False, num_classes=1)
    model.load_state_dict(state_bare, strict=True)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FilePathDataset(Dataset):
    """Load images from OpenFake CSV samples."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, model_name = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (384, 384))
        return CF_TRANSFORM(img), label, model_name



def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    models = [b[2] for b in batch]
    return imgs, labels, models


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def _run_inference(loader, model, device):
    all_logits, all_labels, all_models = [], [], []
    with torch.no_grad():
        for imgs, labels, models in tqdm(loader):
            imgs = imgs.to(device)
            logits = model(imgs).squeeze(1).cpu()
            all_logits.append(logits)
            all_labels.append(labels)
            all_models.extend(models)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy(), np.array(all_models)


def _logits_to_pred_array(logits):
    """Return (N,2) array compatible with compute_per_model_metrics."""
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    return np.stack([1 - probs, probs], axis=1)


def _roc_auc(labels, probs):
    finite = np.isfinite(probs)
    if np.any(finite) and len(np.unique(labels[finite])) > 1:
        return float(roc_auc_score(labels[finite], probs[finite]))
    return None


def _write_outputs(out_csv: str, result_df: pd.DataFrame, test_csv: str, rows: int, roc_auc: float | None) -> None:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out, index=False)
    with open(str(out) + ".json", "w") as f:
        json.dump(
            {
                "checkpoint": "OwensLab/commfor-model-384",
                "test_csv": test_csv,
                "rows": rows,
                "roc_auc": roc_auc,
                "roc_AUC": roc_auc,
                "auc_roc": roc_auc,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )


def evaluate_cf_on_csv(test_csv: str, out_csv: str, batch_size: int = 64,
                        num_workers: int = 4, device: str | None = None) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_cf_model(device)
    dataset = FilePathDataset(samples_from_csv(test_csv))

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device == "cuda"))

    logits, labels, models = _run_inference(loader, model, device)
    preds_2col = _logits_to_pred_array(logits)
    result_df = compute_per_model_metrics(preds_2col, labels, models)
    roc_auc = _roc_auc(labels, preds_2col[:, 1])

    _write_outputs(out_csv, result_df, test_csv, len(dataset), roc_auc)
    print(f"  wrote: {out_csv}  ({len(result_df)} model rows)")
    return result_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", action="append", required=True)
    ap.add_argument("--out_dir", default="results/cf_pretrained")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Load model once
    model = load_cf_model(device)

    for csv_path in args.test_csv:
        stem = Path(csv_path).stem
        out_csv = str(out / f"{stem}.csv")
        print(f"\n--- Evaluating {stem} ---")
        dataset = FilePathDataset(samples_from_csv(csv_path))
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            collate_fn=_collate, pin_memory=(device == "cuda"))
        logits, labels, models = _run_inference(loader, model, device)
        preds_2col = _logits_to_pred_array(logits)
        result_df = compute_per_model_metrics(preds_2col, labels, models)
        roc_auc = _roc_auc(labels, preds_2col[:, 1])
        _write_outputs(out_csv, result_df, csv_path, len(dataset), roc_auc)
        print(f"  wrote: {out_csv}  ({len(result_df)} model rows)")


if __name__ == "__main__":
    main()
