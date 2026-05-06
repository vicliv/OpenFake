#!/usr/bin/env python3
"""Build OpenFake evaluation CSVs from Hugging Face datasets.

The script downloads/materializes ComplexDataLab/OpenFake into a user-provided
directory, then writes CSVs using the schema consumed by the experiment code:

    file_name,prompt,label,model,type,release_date

It does not assume a pre-existing OpenFake image tree. External real OOD
datasets used by the paper split, DOCCI and ImageNet, are passed explicitly.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import get_dataset_split_names, load_dataset
from PIL import Image

SCHEMA = ["file_name", "prompt", "label", "model", "type", "release_date"]
PEXELS_MIN_ID = 1_872_218
PEXELS_MAX_ID = 2_438_962

HELD_OUT_MODELS = {
    "gpt-image-1.5", "gpt-image-2", "nano-banana-pro", "flux.2-klein-9b", "sora-2",
    "z-image-turbo", "illustrious", "lumina-17-2-25", "aurora-20-1-25",
    "frames-23-1-25", "halfmoon-4-4-25", "recraft-v2", "recraft-v3",
    "ideogram-2.0", "midjourney-7", "veo-3", "seedream-v5.0", "ernie-image-turbo",
    "ernie-image", "wan-video-2.5",
}
HELD_OUT_MODEL_PARTS = tuple(sorted(HELD_OUT_MODELS, key=len, reverse=True))
HELD_OUT_MODEL_FAMILIES = {"z-image": "z-image-turbo"}

TEST_INDIST_PER_MODEL = 500
TEST_INDIST_MIN_POOL = 2000
IMAGE_KEYS = ("image", "img", "jpg", "png")
PATH_KEYS = ("file_name", "filename", "image_path", "path")
PROMPT_KEYS = ("prompt", "generated_prompt", "caption", "text")
LABEL_KEYS = ("label", "labels", "class", "target", "is_fake")
MODEL_KEYS = ("model", "generator", "model_name", "source")
TYPE_KEYS = ("type", "format", "media_type")
DATE_KEYS = ("release_date", "timestamp", "post_date", "date")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def sidecar(path: Path, rows: int, args: argparse.Namespace) -> None:
    with open(str(path) + ".json", "w") as f:
        json.dump(
            {
                "script": "build_of_splits.py",
                "rows": rows,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "git_sha": git_sha(),
                "dataset_name": args.openfake_dataset,
                "configs": args.openfake_configs,
                "splits": args.openfake_splits,
                "held_out_models": sorted(HELD_OUT_MODELS),
            },
            f,
            indent=2,
        )


def first_present(row: dict, keys: Iterable[str], default: str = ""):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def label_str(value) -> str:
    if isinstance(value, str):
        return "fake" if value.strip().lower() in {"fake", "synthetic", "full_synthetic", "1", "tampered", "true"} else "real"
    return "fake" if int(value) == 1 else "real"


def save_image(value, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        img = value
    elif isinstance(value, dict) and "bytes" in value:
        from io import BytesIO

        img = Image.open(BytesIO(value["bytes"]))
    else:
        img = Image.open(value)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(out_path)
    return out_path


def row_to_record(row: dict, config: str, split: str, index: int, image_root: Path) -> dict:
    image_value = first_present(row, IMAGE_KEYS, None)
    path_value = first_present(row, PATH_KEYS, "")
    if image_value is not None:
        out_path = image_root / config / split / f"{index:09d}.jpg"
        if not out_path.exists():
            save_image(image_value, out_path)
        file_name = str(out_path)
    elif path_value:
        file_name = str(path_value)
    else:
        available = ", ".join(sorted(row.keys()))
        raise KeyError(f"No image or path column found. Available columns: {available}")

    label_value = first_present(row, LABEL_KEYS, "")
    if label_value == "":
        raise KeyError(f"No label column found for {config}/{split}")

    model = str(first_present(row, MODEL_KEYS, config) or config)
    media_type = str(first_present(row, TYPE_KEYS, "image") or "image")
    return {
        "file_name": file_name,
        "prompt": str(first_present(row, PROMPT_KEYS, "") or ""),
        "label": label_str(label_value),
        "model": model,
        "type": media_type,
        "release_date": str(first_present(row, DATE_KEYS, "") or "")[:10],
    }


def materialize_openfake(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Download OpenFake configs/splits and return normalized DataFrames."""
    image_root = Path(args.openfake_image_dir)
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)

    for config in args.openfake_configs:
        try:
            available_splits = set(get_dataset_split_names(args.openfake_dataset, config, cache_dir=args.hf_cache_dir))
        except Exception as exc:
            raise RuntimeError(f"Could not inspect {args.openfake_dataset} config {config!r}: {exc}") from exc

        for split in args.openfake_splits:
            if split not in available_splits:
                print(f"[WARN] {args.openfake_dataset}/{config}: split {split!r} is unavailable; skipping.")
                continue
            print(f"Loading {args.openfake_dataset}/{config}:{split}...")
            ds = load_dataset(
                args.openfake_dataset,
                config,
                split=split,
                cache_dir=args.hf_cache_dir,
                streaming=args.streaming,
            )
            rows = []
            for index, row in enumerate(ds):
                rows.append(row_to_record(row, config, split, index, image_root))
                if args.max_rows_per_split and len(rows) >= args.max_rows_per_split:
                    break
            if rows:
                frames[split].append(pd.DataFrame(rows, columns=SCHEMA))

    out = {split: pd.concat(parts, ignore_index=True) for split, parts in frames.items() if parts}
    if not out:
        raise RuntimeError("No OpenFake rows were loaded. Check --openfake-configs and --openfake-splits.")
    return out


def pexels_mask(paths: pd.Series, labels: pd.Series) -> pd.Series:
    def is_pexels(path: str) -> bool:
        stem = Path(str(path)).stem
        return stem.isdigit() and PEXELS_MIN_ID <= int(stem) <= PEXELS_MAX_ID

    return labels.str.lower().eq("real") & paths.map(is_pexels)


def normalize_main_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in SCHEMA:
        if col not in df:
            df[col] = ""
    df = df[SCHEMA].fillna("")
    pmask = pexels_mask(df["file_name"], df["label"])
    df.loc[pmask, "model"] = "pexels"
    df.loc[pmask, "type"] = "real"
    laion_mask = df["label"].str.lower().eq("real") & df["model"].str.lower().eq("real")
    df.loc[laion_mask, "model"] = "laion"
    return df


def held_out_match(model: str) -> str | None:
    name = str(model).strip().lower()
    if not name:
        return None
    for held_out in HELD_OUT_MODEL_PARTS:
        if held_out in name:
            return held_out
    for family, held_out in HELD_OUT_MODEL_FAMILIES.items():
        if family in name:
            return held_out
    return None


def split_held_out_fake(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = df["model"].fillna("").astype(str)
    matches = models.map(held_out_match)
    normalized = models.map(lambda m: str(m).strip().lower())
    exact_held = matches.notna() & normalized.eq(matches)
    similar_held = matches.notna() & ~exact_held
    fake_mask = df["label"].str.lower().eq("fake")
    return df[matches.isna()].copy(), df[exact_held & fake_mask].copy(), df[similar_held].copy()


def report_dropped_similar_models(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        return
    matches = df["model"].fillna("").astype(str).map(held_out_match)
    print(f"Dropping held-out-similar models from {source}:")
    for model, count in df["model"].value_counts().head(20).items():
        match = matches[df["model"].eq(model)].iloc[0]
        print(f"  {model}: {count:,} (matched {match})")


def build_test_indist_v2(pool: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    fake = pool[pool["label"].str.lower().eq("fake")].copy()
    real = pool[pool["label"].str.lower().eq("real")].copy()
    qualifying = fake["model"].value_counts().loc[lambda s: s >= TEST_INDIST_MIN_POOL].index.tolist()

    test_fake_parts: list[pd.DataFrame] = []
    train_fake_parts: list[pd.DataFrame] = []
    for model in qualifying:
        rows = fake[fake["model"].eq(model)]
        idx = rows.index.tolist()
        rng.shuffle(idx)
        test_fake_parts.append(fake.loc[idx[:TEST_INDIST_PER_MODEL]])
        train_fake_parts.append(fake.loc[idx[TEST_INDIST_PER_MODEL:]])

    train_fake = pd.concat(train_fake_parts + [fake[~fake["model"].isin(qualifying)]], ignore_index=True)
    n_test_fake = sum(len(p) for p in test_fake_parts)
    real_idx = real.index.tolist()
    rng.shuffle(real_idx)
    test_real = real.loc[real_idx[:n_test_fake]]
    train_real = real.loc[real_idx[n_test_fake:]]
    return pd.concat(test_fake_parts + [test_real], ignore_index=True), pd.concat([train_fake, train_real], ignore_index=True)


def balance_train_real(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    n_fake = (df["label"].str.lower() == "fake").sum()
    n_real = (df["label"].str.lower() == "real").sum()
    if n_real <= n_fake:
        return df
    laion_mask = df["label"].str.lower().eq("real") & df["model"].eq("laion")
    excess = n_real - n_fake
    laion_idx = df.index[laion_mask].tolist()
    rng = random.Random(seed)
    rng.shuffle(laion_idx)
    return df[~df.index.isin(set(laion_idx[: min(excess, len(laion_idx))]))].reset_index(drop=True)


def list_imagenet(imagenet_dir: Path, limit: int) -> list[str]:
    files = []
    for cls in sorted(p for p in imagenet_dir.iterdir() if p.is_dir()):
        for p in sorted(cls.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                files.append(str(p))
                if len(files) >= limit:
                    return files
    return files


def real_ood(n: int, seed: int, docci_dir: Path, imagenet_dir: Path) -> pd.DataFrame:
    if not docci_dir.exists():
        raise FileNotFoundError(f"DOCCI image directory not found: {docci_dir}")
    if not imagenet_dir.exists():
        raise FileNotFoundError(f"ImageNet train directory not found: {imagenet_dir}")
    rng = random.Random(seed)
    docci = sorted(str(p) for p in docci_dir.glob("*.jpg"))
    n_docci = min(n // 2, len(docci))
    n_img = n - n_docci
    imagenet = list_imagenet(imagenet_dir, n_img * 4)
    if len(docci) < n_docci or len(imagenet) < n_img:
        raise RuntimeError(f"Not enough OOD real images: docci={len(docci)} imagenet={len(imagenet)} need={n}")
    rng.shuffle(docci)
    rng.shuffle(imagenet)
    rows = [{"file_name": p, "prompt": "", "label": "real", "model": "docci", "type": "real", "release_date": ""} for p in docci[:n_docci]]
    rows += [{"file_name": p, "prompt": "", "label": "real", "model": "imagenet", "type": "real", "release_date": ""} for p in imagenet[:n_img]]
    return pd.DataFrame(rows, columns=SCHEMA)


def max_ood_size(docci_count: int, imagenet_count: int) -> int:
    lo, hi = 0, docci_count + imagenet_count
    while lo < hi:
        mid = (lo + hi + 1) // 2
        n_docci = min(mid // 2, docci_count)
        n_img = mid - n_docci
        if n_img <= imagenet_count:
            lo = mid
        else:
            hi = mid - 1
    return lo


def trim_held_fake_largest_models(df: pd.DataFrame, target_rows: int, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    if len(df) <= target_rows:
        return df.copy(), {}
    rng = random.Random(seed)
    counts = df["model"].value_counts().sort_values(ascending=False)
    to_remove = len(df) - target_rows
    keep_counts: dict[str, int] = {}
    for model, count in counts.items():
        drop = min(count, to_remove)
        keep_counts[model] = count - drop
        to_remove -= drop
    kept_parts, removed = [], {}
    for model, count in counts.items():
        keep_n = keep_counts[model]
        if keep_n <= 0:
            removed[model] = count
            continue
        rows = df[df["model"].eq(model)]
        idx = rows.index.tolist()
        rng.shuffle(idx)
        kept_parts.append(df.loc[idx[:keep_n]])
        if count - keep_n > 0:
            removed[model] = count - keep_n
    return pd.concat(kept_parts, ignore_index=True), removed


def parse_ood_counts(err: RuntimeError) -> tuple[int, int] | None:
    m = re.search(r"docci=(\d+)\s+imagenet=(\d+)\s+need=(\d+)", str(err))
    return (int(m.group(1)), int(m.group(2))) if m else None


def update_counts(summary: dict, split: str, df: pd.DataFrame) -> None:
    s = summary[split]
    s["rows"] += len(df)
    s["labels"].update(df["label"].fillna("").astype(str))
    s["models"].update(df["model"].fillna("").astype(str))


def format_summary(summary: dict) -> str:
    lines = []
    for split, s in summary.items():
        lines += [f"=== {split} ===", f"rows: {s['rows']:,}", "labels:"]
        lines += [f"  {k}: {v:,}" for k, v in sorted(s["labels"].items())]
        lines.append("models:")
        lines += [f"  {k}: {v:,}" for k, v in s["models"].most_common()]
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--openfake-dataset", default="ComplexDataLab/OpenFake")
    ap.add_argument("--openfake-configs", nargs="+", default=["opensource", "reddit", "inpainting"])
    ap.add_argument("--openfake-splits", nargs="+", default=["train", "test"])
    ap.add_argument("--openfake-image-dir", required=True, help="Directory where HF image payloads are materialized.")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--docci-dir", required=True, help="Directory containing DOCCI images for real OOD rows.")
    ap.add_argument("--imagenet-dir", required=True, help="ImageNet train directory with class subdirectories.")
    ap.add_argument("--streaming", action="store_true", help="Stream rows from the Hub while materializing images.")
    ap.add_argument("--max-rows-per-split", type=int, default=None, help="Debug limit; omit for full split creation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    results_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary = defaultdict(lambda: {"rows": 0, "labels": Counter(), "models": Counter()})

    frames = {name: normalize_main_frame(df) for name, df in materialize_openfake(args).items()}
    train_like_parts = []
    held_parts = []
    for split_name, frame in frames.items():
        non_held, held, dropped = split_held_out_fake(frame)
        report_dropped_similar_models(dropped, split_name)
        train_like_parts.append(non_held)
        if not held.empty:
            held_parts.append(held)

    print("Building OpenFake in-distribution and train splits...")
    pool = pd.concat(train_like_parts, ignore_index=True)
    test_indist, train_df = build_test_indist_v2(pool, args.seed)
    train_df = balance_train_real(train_df, args.seed)

    train_path = out_dir / "of_train_v2.csv"
    train_df.to_csv(train_path, index=False)
    update_counts(summary, "of_train_v2", train_df)
    sidecar(train_path, len(train_df), args)

    indist_path = out_dir / "of_test_indist_v2.csv"
    test_indist.to_csv(indist_path, index=False)
    update_counts(summary, "of_test_indist_v2", test_indist)
    sidecar(indist_path, len(test_indist), args)

    held_fake = pd.concat([p for p in held_parts if not p.empty], ignore_index=True) if held_parts else pd.DataFrame(columns=SCHEMA)
    try:
        ood_real = real_ood(len(held_fake), args.seed, Path(args.docci_dir), Path(args.imagenet_dir))
    except RuntimeError as err:
        counts = parse_ood_counts(err)
        if counts is None:
            raise
        max_fake = max_ood_size(*counts)
        if max_fake <= 0:
            raise RuntimeError(f"No feasible OOD set with available real images: docci={counts[0]} imagenet={counts[1]}") from err
        trimmed_fake, removed_by_model = trim_held_fake_largest_models(held_fake, max_fake, args.seed)
        print(f"Insufficient OOD real images; trimming held-out fake rows from {len(held_fake):,} to {len(trimmed_fake):,}.")
        for model, removed in sorted(removed_by_model.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  removed {removed:,} from {model}")
        held_fake = trimmed_fake
        ood_real = real_ood(len(held_fake), args.seed, Path(args.docci_dir), Path(args.imagenet_dir))

    ood = pd.concat([held_fake, ood_real], ignore_index=True)
    ood_path = out_dir / "of_test_ood_models_v2.csv"
    ood.to_csv(ood_path, index=False)
    update_counts(summary, "of_test_ood_models_v2", ood)
    sidecar(ood_path, len(ood), args)

    text = format_summary(summary)
    print(text)
    (results_dir / "split_summary_v2.txt").write_text(text)


if __name__ == "__main__":
    main()
