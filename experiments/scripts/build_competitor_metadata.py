#!/usr/bin/env python3
"""Normalize competitor eval metadata into OpenFake CSV schema.

Expects images to already be materialized on disk by download_eval_sets.py.
Run download_eval_sets.py first.

Native fields:
  CF Eval    : label (0/1 int or "0"/"1" str), model_name, prompt, image_name
  So-Fake-OOD: label (0/1 int), generator, filename
  Semi-Truths: __key__ (e.g. StableDiffusion_v5/instance_000_...), png image
  GenImage   : from --genimage-root/metadata.csv, type='genimage'
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import os

import pandas as pd
from datasets import load_dataset

SCHEMA = ["file_name", "prompt", "label", "model", "type", "release_date"]


def _label_str(value) -> str:
    if isinstance(value, str):
        return "fake" if value.strip().lower() in {"fake", "synthetic", "1", "tampered"} else "real"
    return "fake" if int(value) == 1 else "real"


def side_summary(name: str, df: pd.DataFrame, results_dir: Path) -> None:
    text = [f"rows: {len(df):,}", "labels:"]
    text += [f"  {k}: {v:,}" for k, v in df["label"].value_counts().items()]
    text += ["models:"]
    text += [f"  {k}: {v:,}" for k, v in df["model"].value_counts().head(100).items()]
    (results_dir / f"{name}_summary.txt").write_text("\n".join(text))
    with open(results_dir / f"{name}_summary.json", "w") as f:
        json.dump({"rows": len(df), "timestamp": datetime.utcnow().isoformat() + "Z"}, f, indent=2)


def build_cf(out_dir: Path, results_dir: Path, competitor_root: Path) -> None:
    img_dir = competitor_root / "cf_eval" / "images"
    assert img_dir.exists(), (
        f"CF images not found at {img_dir}. Run download_eval_sets.py first."
    )
    print("Building cf_eval.csv from on-disk CF images + HF streaming metadata...")
    ds = load_dataset("OwensLab/CommunityForensics-Eval", split="CompEval", streaming=True)
    rows = []
    for r in ds:
        name = r.get("image_name") or ""
        path = img_dir / name
        lab = _label_str(r.get("label", 0))
        model = r.get("model_name") or r.get("architecture") or "unknown_cf"
        prompt = r.get("prompt") or ""
        rows.append({
            "file_name": str(path),
            "prompt": prompt,
            "label": lab,
            "model": model,
            "type": "cf",
            "release_date": "",
        })
    df = pd.DataFrame(rows, columns=SCHEMA)
    out = out_dir / "cf_eval.csv"
    df.to_csv(out, index=False)
    side_summary("cf_eval", df, results_dir)
    print(f"  cf_eval.csv: {len(df):,} rows")


def build_sofake(out_dir: Path, results_dir: Path, competitor_root: Path) -> None:
    img_dir = competitor_root / "sofake_ood"
    assert img_dir.exists(), (
        f"So-Fake images not found at {img_dir}. Run download_eval_sets.py first."
    )
    print("Building sofake_ood.csv from on-disk So-Fake images + HF streaming metadata...")
    ds = load_dataset("saberzl/So-Fake-OOD", split="test", streaming=True)
    rows = []
    for r in ds:
        name = r.get("filename") or ""
        path = img_dir / name
        lab = _label_str(r.get("label", 0))
        model = r.get("generator") or "unknown_sofake"
        rows.append({
            "file_name": str(path),
            "prompt": "",
            "label": lab,
            "model": model,
            "type": "sofake",
            "release_date": "",
        })
    df = pd.DataFrame(rows, columns=SCHEMA)
    out = out_dir / "sofake_ood.csv"
    df.to_csv(out, index=False)
    side_summary("sofake_ood", df, results_dir)
    print(f"  sofake_ood.csv: {len(df):,} rows")


def build_semitruths(out_dir: Path, results_dir: Path, competitor_root: Path) -> None:
    img_dir = competitor_root / "semitruths_eval"
    assert img_dir.exists(), (
        f"Semi-Truths images not found at {img_dir}. Run download_eval_sets.py first."
    )
    print("Building semitruths_eval.csv from on-disk Semi-Truths images + HF streaming metadata...")
    ds = load_dataset("semi-truths/Semi-Truths-Evalset", split="train", streaming=True)
    rows = []
    for r in ds:
        key = r.get("__key__", "")
        url = r.get("__url__", "")
        if "mask" in url:
            continue
        safe_key = key.replace("/", "__")
        path = img_dir / f"{safe_key}.png"
        # Match the original Semi-Truths SwinV2 eval script:
        # only original images are real; editing/inpainting rows are fake.
        lab = "real" if "original" in url else "fake"
        parts = str(key).split("/")
        model = parts[0] if lab == "fake" and parts else "real"
        rows.append({
            "file_name": str(path),
            "prompt": "",
            "label": lab,
            "model": model,
            "type": "semitruths",
            "release_date": "",
        })
    df = pd.DataFrame(rows, columns=SCHEMA)
    out = out_dir / "semitruths_eval.csv"
    df.to_csv(out, index=False)
    side_summary("semitruths_eval", df, results_dir)
    print(f"  semitruths_eval.csv: {len(df):,} rows")


def build_genimage(out_dir: Path, results_dir: Path, genimage_root: Path) -> None:
    meta_path = genimage_root / "metadata.csv"
    assert meta_path.exists(), f"GenImage metadata not found: {meta_path}"
    df = pd.read_csv(meta_path, dtype=str, low_memory=False, on_bad_lines="skip").fillna("")
    df = df[df["type"].str.lower() == "genimage"].copy()
    df["file_name"] = df["file_name"].map(
        lambda p: str(genimage_root / p) if not str(p).startswith("/") else p
    )
    # subsample 25,000 randomly
    if len(df) > 25_000:
        df = df.sample(25_000, random_state=42).reset_index(drop=True)
    out = df.reindex(columns=SCHEMA).fillna("")
    out.to_csv(out_dir / "genimage_test.csv", index=False)
    side_summary("genimage_test", out, results_dir)
    print(f"  genimage_test.csv: {len(out):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--competitor-root", required=True, help="Root created by download_eval_sets.py.")
    ap.add_argument("--genimage-root", default=None, help="Local GenImage root containing metadata.csv.")
    ap.add_argument("--datasets", nargs="+",
                    choices=["cf", "sofake", "semitruths", "genimage", "all"],
                    default=["all"])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    competitor_root = Path(args.competitor_root)
    genimage_root = Path(args.genimage_root) if args.genimage_root else None
    out_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    do = set(args.datasets)
    if "all" in do or "cf" in do:
        build_cf(out_dir, results_dir, competitor_root)
    if "all" in do or "sofake" in do:
        build_sofake(out_dir, results_dir, competitor_root)
    if "all" in do or "semitruths" in do:
        build_semitruths(out_dir, results_dir, competitor_root)
    if "all" in do or "genimage" in do:
        if genimage_root is None:
            raise ValueError("--genimage-root is required when building genimage metadata.")
        build_genimage(out_dir, results_dir, genimage_root)


if __name__ == "__main__":
    main()
