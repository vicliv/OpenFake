#!/usr/bin/env python3
"""CLIP-D (clipdet_latent10k_plus) and Corvi2023 baseline evaluator.

Wraps the model definitions and weight-loading in the directory passed by
--baseline-dir.

Preprocessing (from reference main.py / utils/processing.py):
  Corvi2023 (arch=res50nodown):
    - patch_size=Force224 → Resize((224, 224), BICUBIC), matching the reference
      DataLoader path's fixed-size batching behavior
    - norm_type=resnet → ToTensor + Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

  clipdet_latent10k_plus (arch=opencliplinearnext_clipL14commonpool):
    - patch_size=Clip224 → Resize(224, BICUBIC) + CenterCrop(224)
    - norm_type=clip → ToTensor + Normalize(mean=(0.48145466,0.4578275,0.40821073),
                                            std=(0.26862954,0.26130258,0.27577711))

Weights (per preflight):
  $SCRATCH/model_weights/CLIP/Corvi2023/weights.pth         (282 MB)
  $SCRATCH/model_weights/CLIP/clipdet_latent10k_plus/weights.pth  (5 KB linear probe)

Output: results/baselines/clip/<model_name>_<test_set_stem>.csv
CLI: --test_csv <path> --out_csv <path>  (can repeat --test_csv)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR.parent))
from training.metrics import compute_per_model_metrics  # noqa: E402

# ---- Weight paths -----------------------------------------------------------
WEIGHTS_ROOT = Path("model_weights")
MODELS = {
    "Corvi2023": {
        "arch": "res50nodown",
        "norm_type": "resnet",
        "patch_size": "Force224",
        "weights": "CLIP/Corvi2023/weights.pth",
    },
    "clipdet_latent10k_plus": {
        "arch": "opencliplinearnext_clipL14commonpool",
        "norm_type": "clip",
        "patch_size": "Clip224",
        "weights": "CLIP/clipdet_latent10k_plus/weights.pth",
    },
}


def _build_transform(patch_size, norm_type):
    steps = []
    if patch_size == "Force224":
        steps.append(Resize((224, 224), interpolation=InterpolationMode.BICUBIC))
    elif patch_size == "Clip224":
        steps.append(Resize(224, interpolation=InterpolationMode.BICUBIC))
        steps.append(CenterCrop((224, 224)))
    steps.append(make_normalize(norm_type))
    return Compose(steps)


def _load_model(name: str, device: str):
    cfg = MODELS[name]
    from networks import create_architecture, load_weights

    model = create_architecture(cfg["arch"])
    model = load_weights(model, str(WEIGHTS_ROOT / cfg["weights"]))
    return model.to(device).eval(), _build_transform(cfg["patch_size"], cfg["norm_type"])


class CSVDataset(Dataset):
    """Loads PIL images from an OpenFake-schema CSV."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["file_name"].astype(str).tolist()
        self.labels = df["label"].str.lower().eq("fake").astype(int).tolist()
        self.models = df["model"].astype(str).tolist() if "model" in df.columns else [""] * len(df)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), self.labels[idx], self.models[idx]


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    models = [b[2] for b in batch]
    return imgs, labels, models


def _run_single_model(model_name: str, test_csv: str, out_csv: str,
                       device: str, batch_size: int = 64, num_workers: int = 4):
    df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
    model, transform = _load_model(model_name, device)
    ds = CSVDataset(df, transform)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device == "cuda"))

    all_logits, all_labels, all_models = [], [], []
    with torch.no_grad():
        for imgs, labels, models in tqdm(loader, desc=f"{model_name}/{Path(test_csv).stem}"):
            imgs = imgs.to(device)
            out = model(imgs).cpu()
            # Reference outputs shape (N,1) or (N,2); collapse to single score
            if out.shape[1] == 1:
                score = out[:, 0]
            else:
                score = out[:, 1] - out[:, 0]
            all_logits.append(score)
            all_labels.append(labels)
            all_models.extend(models)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    model_names = np.array(all_models)

    # Build (N,2) array for compute_per_model_metrics
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds_2col = np.stack([1 - probs, probs], axis=1)
    result_df = compute_per_model_metrics(preds_2col, labels, model_names)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    with open(out_csv + ".json", "w") as f:
        json.dump({"model": model_name, "test_csv": test_csv,
                   "rows": len(ds), "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"  wrote: {out_csv}  ({len(result_df)} model rows)")
    return result_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", action="append", required=True)
    ap.add_argument("--out_dir", default="results/baselines/clip")
    ap.add_argument("--baseline_dir", required=True, help="Path to the CLIP_baseline reference code.")
    ap.add_argument("--weights_root", required=True, help="Root containing the model_weights tree.")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    global WEIGHTS_ROOT, make_normalize
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {baseline_dir}")
    sys.path.insert(0, str(baseline_dir))
    from utils.processing import make_normalize as _make_normalize

    make_normalize = _make_normalize
    WEIGHTS_ROOT = Path(args.weights_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        if model_name not in MODELS:
            print(f"[WARN] Unknown model {model_name}, skipping.")
            continue
        for csv_path in args.test_csv:
            stem = Path(csv_path).stem
            out_csv = str(out / f"{model_name}_{stem}.csv")
            print(f"\n--- {model_name} on {stem} ---")
            _run_single_model(model_name, csv_path, out_csv, device,
                               args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
