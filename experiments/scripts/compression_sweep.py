#!/usr/bin/env python3
"""JPEG compression robustness sweep.

Evaluates a checkpoint at quality levels: none (no compression), 90, 75, 50, 25.
For each quality, applies in-memory JPEG encode → decode before model inference.

Usage:
  python compression_sweep.py \\
    --checkpoint /path/to/checkpoint \\
    --test_csv /path/to/test.csv \\
    --out_csv results/compression/of_v2_ood.csv

Output: one row per (quality, model) combination.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

BASE_SWINV2 = "microsoft/swinv2-small-patch4-window16-256"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import OpenFakeDataset  # noqa: E402
from data.collator import OpenFakeCollator  # noqa: E402
from training.metrics import compute_per_model_metrics  # noqa: E402


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JpegDataset(OpenFakeDataset):
    """Applies in-memory JPEG compression after loading each image."""

    def __init__(self, samples, quality: int | None):
        super().__init__(samples, transform=None)
        self.quality = quality

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self.quality is not None:
            item["image"] = jpeg_compress(item["image"], self.quality)
        return item


def samples_from_csv(test_csv: str):
    df = pd.read_csv(test_csv, dtype=str, low_memory=False).fillna("")
    labels = df["label"].str.lower().eq("fake").astype(int)
    models = df["model"].astype(str) if "model" in df.columns else pd.Series([""] * len(df))
    return list(zip(df["file_name"].astype(str), labels, models))


def run_quality(ckpt: str, samples, quality: int | None,
                processor, model, device: str,
                batch_size: int, num_workers: int) -> pd.DataFrame:
    ds = JpegDataset(samples, quality)
    # model_names live in the dataset samples; collect them in order
    all_model_names = [s[2] if len(s) > 2 else "" for s in samples]

    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                        collate_fn=OpenFakeCollator(processor),
                        pin_memory=(device == "cuda"))

    # Collect predictions in dataset order (DataLoader preserves order with shuffle=False)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"q={quality or 'none'}"):
            all_labels.append(batch.pop("labels").cpu())
            batch = {k: v.to(device) for k, v in batch.items()}
            all_logits.append(model(**batch).logits.cpu())
    preds = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    model_arr = np.array(all_model_names[:len(labels)])
    df = compute_per_model_metrics(preds, labels, model_arr)
    df["quality"] = "none" if quality is None else str(quality)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt_path = Path(args.checkpoint)
    proc_src = args.checkpoint
    if (ckpt_path.is_absolute() or ckpt_path.exists()) and not (ckpt_path / "preprocessor_config.json").exists():
        proc_src = BASE_SWINV2
    try:
        processor = AutoImageProcessor.from_pretrained(proc_src, use_fast=True)
    except (OSError, TypeError):
        processor = AutoImageProcessor.from_pretrained(BASE_SWINV2, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained(args.checkpoint).to(device).eval()

    samples = samples_from_csv(args.test_csv)
    qualities = [None, 90, 75, 50, 25]  # None = no compression baseline

    rows = []
    for q in qualities:
        df = run_quality(args.checkpoint, samples, q, processor, model,
                         device, args.batch_size, args.num_workers)
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    with open(str(out) + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint, "test_csv": args.test_csv,
                   "rows": len(samples), "qualities": [str(q) for q in qualities],
                   "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"Wrote {out} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
