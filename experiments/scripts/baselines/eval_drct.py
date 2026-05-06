#!/usr/bin/env python3
"""DRCT-2M (SDv2 checkpoint) baseline evaluator.

Reference code is passed with --baseline-dir.
Checkpoints:
  convnext_base_in22k_224_drct_amp_crop  → 16_acc0.9993.pth  (primary, higher acc)
  clip-ViT-L-14_224_drct_amp_crop       → last_acc0.9112.pth (fallback)

Preprocessing (from DRCT/test.py create_val_transforms):
  - LongestMaxSize(max_size=224)
  - PadIfNeeded(min_height=224, min_width=224, border_mode=cv2.BORDER_CONSTANT, value=0)
  - Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
  - ToTensorV2()

CLI: --test_csv <path> --out_csv <path>
     --model_name convnext_base_in22k  (default) or clip-ViT-L-14
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.models.convnext import checkpoint_filter_fn as convnext_checkpoint_filter_fn

ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THIS_DIR.parent))
from training.metrics import compute_per_model_metrics  # noqa: E402

WEIGHTS_ROOT = Path("model_weights")

# Map model_name -> (timm model name, checkpoint filename, embedding size)
CHECKPOINT_MAP = {
    "convnext_base_in22k": (
        "convnext_base_in22k",
        "DRCT/pretrained/DRCT-2M/sdv2/convnext_base_in22k_224_drct_amp_crop/16_acc0.9993.pth",
        1024,
    ),
    "clip-ViT-L-14": (
        "clip-ViT-L-14",
        "DRCT/pretrained/DRCT-2M/sdv2/clip-ViT-L-14_224_drct_amp_crop/last_acc0.9112.pth",
        512,
    ),
}
DEFAULT_MODEL = "convnext_base_in22k"

# Preprocessing: replicate DRCT test.py create_val_transforms(size=224)
TRANSFORM = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(min_height=224, min_width=224,
                  border_mode=cv2.BORDER_CONSTANT, fill=0),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def load_model(model_name: str, device: str) -> torch.nn.Module:
    timm_name, ckpt_path, embedding_size = CHECKPOINT_MAP[model_name]
    from network.models import get_models

    model = get_models(
        model_name=timm_name,
        num_classes=2,
        pretrained=False,
        embedding_size=embedding_size,
    )
    state = torch.load(str(WEIGHTS_ROOT / ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    if "convnext" in timm_name:
        inner_prefix = "model."
        inner_state = {
            k[len(inner_prefix):]: v
            for k, v in state.items()
            if k.startswith(inner_prefix)
        }
        inner_state = convnext_checkpoint_filter_fn(inner_state, model.model)
        state = {
            **{k: v for k, v in state.items() if not k.startswith(inner_prefix)},
            **{inner_prefix + k: v for k, v in inner_state.items()},
        }
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
            img = np.array(Image.open(self.paths[idx]).convert("RGB"))
        except Exception:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        tensor = TRANSFORM(image=img)["image"]
        return tensor, self.labels[idx], self.models[idx]


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    models = [b[2] for b in batch]
    return imgs, labels, models


def evaluate_drct(test_csv: str, out_csv: str, model_name: str = DEFAULT_MODEL,
                   batch_size: int = 64, num_workers: int = 4,
                   device: str | None = None) -> pd.DataFrame:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
    model = load_model(model_name, device)
    ds = CSVDataset(df)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=_collate, pin_memory=(device == "cuda"))

    all_probs, all_labels, all_models = [], [], []
    with torch.no_grad():
        for imgs, labels, models in tqdm(loader, desc=Path(test_csv).stem):
            imgs = imgs.to(device)
            logits = model(imgs)
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
        json.dump({"model": f"drct_{model_name}", "test_csv": test_csv,
                   "rows": len(ds), "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"  wrote: {out_csv}  ({len(result_df)} model rows)")
    return result_df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", action="append", required=True)
    ap.add_argument("--out_dir", default="results/baselines/drct")
    ap.add_argument("--baseline_dir", required=True, help="Path to the DRCT reference code.")
    ap.add_argument("--weights_root", required=True, help="Root containing the model_weights tree.")
    ap.add_argument("--model_name", choices=list(CHECKPOINT_MAP.keys()), default=DEFAULT_MODEL)
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
        out_csv = str(out / f"drct_{args.model_name}_{stem}.csv")
        evaluate_drct(csv_path, out_csv, args.model_name, args.batch_size, args.num_workers, device)


if __name__ == "__main__":
    main()
