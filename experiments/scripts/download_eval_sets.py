#!/usr/bin/env python3
"""Download and materialize competitor eval sets to disk.

CF Eval (OwensLab/CommunityForensics-Eval, CompEval split):
  Images are base64/bytes embedded in the parquet. We decode and save each image
  under $SCRATCH/competitor_eval/cf_eval/images/<image_name>.

So-Fake-OOD (saberzl/So-Fake-OOD, test split):
  PIL images in the HF dataset. Saved under $SCRATCH/competitor_eval/sofake_ood/<filename>.

Semi-Truths Eval (semi-truths/Semi-Truths-Evalset, train split):
  WebDataset-style with PNG images and key strings.
  Saved under $SCRATCH/competitor_eval/semitruths_eval/<key>.png.

GenImage is not downloaded by this script. Pass its local root to
build_competitor_metadata.py when creating genimage_test.csv.

Re-run is idempotent: skips files that already exist.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

_MAX_SIDE = 2048  # cap resolution for sofake/semitruths PIL images


def _ext_from_bytes(raw: bytes) -> str:
    if raw[:2] == b"\xff\xd8":
        return ".jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return ".webp"
    if raw[4:8] == b"ftyp":
        return ".avif"
    return ".jpg"


def _save_cf(root: Path) -> dict:
    img_dir = root / "cf_eval" / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    print("Streaming OwensLab/CommunityForensics-Eval CompEval...")
    ds = load_dataset("OwensLab/CommunityForensics-Eval", split="CompEval", streaming=True)
    saved = skipped = errors = 0
    for r in tqdm(ds, desc="cf_eval"):
        raw = r.get("image_data")
        if isinstance(raw, str):
            raw = raw.encode("latin-1")
        # Use image_name if present; otherwise derive extension from magic bytes.
        name = r.get("image_name") or f"cf_{saved + skipped:08d}{_ext_from_bytes(raw)}"
        out = img_dir / name
        if out.exists():
            skipped += 1
            continue
        try:
            # Write raw compressed bytes directly — no decode needed, saves
            # hundreds of MB per large image vs going through PIL.
            out.write_bytes(raw)
            saved += 1
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  [WARN] {name}: {e}")
    meta = {"saved": saved, "skipped": skipped, "errors": errors,
            "dir": str(img_dir), "timestamp": datetime.utcnow().isoformat() + "Z"}
    (root / "cf_eval" / "download.json").write_text(json.dumps(meta, indent=2))
    print(f"  CF Eval: saved={saved} skipped={skipped} errors={errors}")
    return meta


def _save_sofake(root: Path) -> dict:
    img_dir = root / "sofake_ood"
    img_dir.mkdir(parents=True, exist_ok=True)
    print("Streaming saberzl/So-Fake-OOD test split...")
    ds = load_dataset("saberzl/So-Fake-OOD", split="test", streaming=True)
    saved = skipped = errors = 0
    for r in tqdm(ds, desc="sofake_ood"):
        name = r.get("filename") or f"sofake_{saved + skipped:08d}.png"
        out = img_dir / name
        if out.exists():
            skipped += 1
            continue
        try:
            img = r["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
            img.save(str(out))
            saved += 1
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  [WARN] {name}: {e}")
    meta = {"saved": saved, "skipped": skipped, "errors": errors,
            "dir": str(img_dir), "timestamp": datetime.utcnow().isoformat() + "Z"}
    (img_dir / "download.json").write_text(json.dumps(meta, indent=2))
    print(f"  So-Fake-OOD: saved={saved} skipped={skipped} errors={errors}")
    return meta


def _save_semitruths(root: Path) -> dict:
    img_dir = root / "semitruths_eval"
    img_dir.mkdir(parents=True, exist_ok=True)
    print("Streaming semi-truths/Semi-Truths-Evalset train split...")
    ds = load_dataset("semi-truths/Semi-Truths-Evalset", split="train", streaming=True)
    saved = skipped = errors = 0
    for r in tqdm(ds, desc="semitruths_eval"):
        key = r.get("__key__", f"semitruths_{saved + skipped:08d}")
        if "mask" in (r.get("__url__") or ""):
            skipped += 1
            continue
        safe_key = key.replace("/", "__")
        out = img_dir / f"{safe_key}.png"
        if out.exists():
            skipped += 1
            continue
        try:
            img = r.get("png")
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
            img.save(str(out))
            saved += 1
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  [WARN] {key}: {e}")
    meta = {"saved": saved, "skipped": skipped, "errors": errors,
            "dir": str(img_dir), "timestamp": datetime.utcnow().isoformat() + "Z"}
    (img_dir / "download.json").write_text(json.dumps(meta, indent=2))
    print(f"  Semi-Truths Eval: saved={saved} skipped={skipped} errors={errors}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Directory where downloaded competitor images are stored.")
    ap.add_argument("--datasets", nargs="+", choices=["cf", "sofake", "semitruths", "all"],
                    default=["all"])
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    do = set(args.datasets)
    if "all" in do or "cf" in do:
        _save_cf(root)
    if "all" in do or "sofake" in do:
        _save_sofake(root)
    if "all" in do or "semitruths" in do:
        _save_semitruths(root)
    print("Done.")


if __name__ == "__main__":
    main()
