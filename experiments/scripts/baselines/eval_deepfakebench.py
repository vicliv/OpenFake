#!/usr/bin/env python3
"""DeepFakeBench EfficientNet-B4 (FF++) baseline evaluator.

Reference code is passed with --baseline-dir.
Checkpoint: $SCRATCH/model_weights/DeepFakeBench/effnb4_best.pth

Preprocessing (from reference efficientnetb4_detector.py __main__ block):
  - Resize to (256, 256)
  - Normalize mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

Architecture: EfficientNetB4 with num_classes=2, inc=3, dropout=False, mode='original'.
Output: softmax[:, 1] → fake probability; threshold 0.5 → fake if > 0.5.

CLI: --test_csv <path> --out_csv <path>
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
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR.parent))
from training.metrics import compute_per_model_metrics  # noqa: E402

WEIGHTS_ROOT = Path("model_weights")
WEIGHT_PATH = Path("DeepFakeBench/effnb4_best.pth")

# Preprocessing matches the reference test harness in efficientnetb4_detector.py
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_model(device: str) -> torch.nn.Module:
    from efficientnetb4 import EfficientNetB4

    config = {"num_classes": 2, "inc": 3, "dropout": False, "mode": "original"}
    model = EfficientNetB4(config)
    state = torch.load(str(WEIGHTS_ROOT / WEIGHT_PATH), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths = df["file_name"].astype(str).tolist()
        self.labels = df["label"].str.lower().eq("fake").astype(int).tolist()
        self.models = df["model"].astype(str).tolist() if "model" in df.columns else [""] * len(df)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        return TRANSFORM(img), self.labels[idx], self.models[idx]


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    models = [b[2] for b in batch]
    return imgs, labels, models


def evaluate_deepfakebench(test_csv: str, out_csv: str,
                            batch_size: int = 64, num_workers: int = 4,
                            device: str | None = None) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
    model = load_model(device)
    ds = CSVDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device == "cuda"))

    all_probs, all_labels, all_models = [], [], []
    with torch.no_grad():
        for imgs, labels, models in tqdm(loader, desc=Path(test_csv).stem):
            imgs = imgs.to(device)
            logits = model.efficientnet(imgs)
            logits = model.last_layer(logits)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu()
            all_probs.append(probs)
            all_labels.append(labels)
            all_models.extend(models)

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    models_np = np.array(all_models)
    preds_2col = np.stack([1 - probs_np, probs_np], axis=1)
    result_df = compute_per_model_metrics(preds_2col, labels_np, models_np)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_csv, index=False)
    with open(out_csv + ".json", "w") as f:
        json.dump({"model": "deepfakebench_effnb4", "test_csv": test_csv,
                   "rows": len(ds), "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"  wrote: {out_csv}  ({len(result_df)} model rows)")
    return result_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", action="append", required=True)
    ap.add_argument("--out_dir", default="results/baselines/ff++")
    ap.add_argument("--baseline_dir", required=True, help="Path to the DeepFakeBench reference code.")
    ap.add_argument("--weights_root", required=True, help="Root containing the model_weights tree.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    global WEIGHTS_ROOT
    baseline_dir = Path(args.baseline_dir)
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Reference dir not found: {baseline_dir}")
    sys.path.insert(0, str(baseline_dir))
    WEIGHTS_ROOT = Path(args.weights_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for csv_path in args.test_csv:
        stem = Path(csv_path).stem
        out_csv = str(out / f"effnb4_{stem}.csv")
        evaluate_deepfakebench(csv_path, out_csv, args.batch_size, args.num_workers, device)


if __name__ == "__main__":
    main()
