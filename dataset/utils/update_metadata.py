"""
Backfill metadata CSV for images that were generated without metadata tracking.

Uses model_registry.json to match filenames to model IDs. Filenames contain
the model ID with "/" replaced by "_", so we match against registry keys.

Handles both old format (hf_{model}_{index}.png) and new format 
(hf_{model}_{date}_{index}.png).

Usage:
    python backfill_metadata.py [--dry-run]
"""

import os
import re
import csv
import json
import sys
import argparse
from collections import Counter
from datetime import datetime
from huggingface_hub import model_info as hf_model_info

STAGING_DIR = "data/staging_images"
METADATA_CSV = os.path.join(STAGING_DIR, "metadata.csv")
REGISTRY_FILE = "model_registry.json"
METADATA_FIELDS = ["filename", "prompt", "label", "model", "type", "release_date", "packaged"]


def parse_args():
    parser = argparse.ArgumentParser(description="Backfill metadata for untracked images.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be written without writing.")
    parser.add_argument("--staging-dir", type=str, default=STAGING_DIR, help="Directory containing images.")
    parser.add_argument("--output", type=str, default=METADATA_CSV, help="Path to metadata CSV.")
    parser.add_argument("--registry", type=str, default=REGISTRY_FILE, help="Path to model registry JSON.")
    return parser.parse_args()


def load_existing_metadata(csv_path):
    """Load already-tracked filenames from the metadata CSV."""
    tracked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tracked.add(row["filename"])
    return tracked


def load_registry(registry_path):
    """Load model registry and return dict of model_id -> entry."""
    if not os.path.exists(registry_path):
        print(f"ERROR: Registry file not found at {registry_path}")
        sys.exit(1)
    with open(registry_path, "r") as f:
        return json.load(f)


def build_model_lookup(registry):
    """
    Build a lookup: safe_model_name -> model_id
    
    Registry keys are like "CompVis/stable-diffusion-v1-4".
    Filenames use "CompVis_stable-diffusion-v1-4".
    
    Sort by length descending so longer (more specific) matches win first.
    e.g. "stable-diffusion-v1-5/stable-diffusion-v1-5" matches before 
    a hypothetical shorter model.
    """
    lookup = {}
    for model_id in registry:
        safe_name = model_id.replace("/", "_")
        lookup[safe_name] = model_id

    # Sort by key length descending for greedy matching
    sorted_lookup = sorted(lookup.items(), key=lambda x: len(x[0]), reverse=True)
    return sorted_lookup


def match_filename(filename, sorted_lookup):
    """
    Match a filename to a model ID using the registry lookup.
    
    Strips "hf_" prefix, then checks if any registry model's safe name
    appears at the start of the remaining string.
    
    Returns (model_id, date_str or None) or (None, None).
    """
    if not filename.startswith("hf_") or not filename.endswith(".png"):
        return None, None

    stem = filename[3:-4]  # strip "hf_" and ".png"

    for safe_name, model_id in sorted_lookup:
        if stem.startswith(safe_name):
            # Everything after the model name
            remainder = stem[len(safe_name):]

            # Remainder should be like "_2026-04-08_4785" or "_1713"
            # Try new format with date first
            date_match = re.match(r'^_(\d{4}-\d{2}-\d{2})_(\d+)$', remainder)
            if date_match:
                return model_id, date_match.group(1)

            # Old format: just _index
            index_match = re.match(r'^_(\d+)$', remainder)
            if index_match:
                return model_id, None

    return None, None


def fetch_release_dates(model_ids):
    """Fetch release dates for a set of model IDs from HuggingFace."""
    dates = {}
    for mid in sorted(model_ids):
        try:
            info = hf_model_info(mid)
            if getattr(info, "created_at", None):
                dates[mid] = info.created_at.strftime("%Y-%m-%d")
            else:
                dates[mid] = "unknown"
            print(f"  {mid}: {dates[mid]}")
        except Exception as e:
            print(f"  WARNING: Could not fetch info for {mid}: {e}")
            dates[mid] = "unknown"
    return dates


def infer_model_type(count):
    """Infer model type from image count per model."""
    if count >= 8000:
        return "Base"
    elif count >= 4000:
        return "Fine-tune"
    else:
        return "LoRA"


def main():
    args = parse_args()
    staging_dir = args.staging_dir
    output_csv = args.output

    # Load registry
    registry = load_registry(args.registry)
    sorted_lookup = build_model_lookup(registry)
    print(f"Loaded {len(sorted_lookup)} models from registry.")

    print(f"\nScanning {staging_dir} for images...")
    all_files = [f for f in os.listdir(staging_dir) if f.endswith(".png")]
    print(f"Found {len(all_files)} total .png files.")

    # Load already-tracked filenames
    existing = load_existing_metadata(output_csv)
    print(f"Already tracked in metadata CSV: {len(existing)} files.")

    # Filter to untracked files only
    untracked = [f for f in all_files if f not in existing]
    print(f"Untracked files to backfill: {len(untracked)}")

    if not untracked:
        print("Nothing to backfill. Exiting.")
        return

    # Match filenames to models
    matched = {}   # filename -> (model_id, date_str or None)
    unmatched = []
    for f in untracked:
        model_id, date_str = match_filename(f, sorted_lookup)
        if model_id:
            matched[f] = (model_id, date_str)
        else:
            unmatched.append(f)

    print(f"\nMatched: {len(matched)} files")
    print(f"Unmatched: {len(unmatched)} files")

    if unmatched:
        print("Examples of unmatched files:")
        for f in unmatched[:10]:
            print(f"  {f}")

    if not matched:
        print("No files matched. Check your registry and filename patterns.")
        return

    # Count images per model
    model_counts = Counter(mid for mid, _ in matched.values())

    # Fetch release dates
    unique_models = set(mid for mid, _ in matched.values())
    print(f"\nFetching release dates for {len(unique_models)} models...")
    release_dates = fetch_release_dates(unique_models)

    # Infer types and print summary
    model_types = {}
    print(f"\nModel summary:")
    for mid, count in model_counts.most_common():
        model_types[mid] = infer_model_type(count)
        print(f"  {mid}: {count} images -> {model_types[mid]}")

    # Build rows
    rows = []
    for filename, (model_id, date_str) in matched.items():
        rows.append({
            "filename": filename,
            "prompt": "",
            "label": "fake",
            "model": model_id,
            "type": model_types[model_id],
            "release_date": release_dates.get(model_id, "unknown"),
            "packaged": False,
        })

    print(f"\nBuilt {len(rows)} metadata rows.")

    if args.dry_run:
        print("\n[DRY RUN] Would write the following (first 5):")
        for row in rows[:5]:
            print(f"  {row}")
        if len(rows) > 5:
            print(f"  ... and {len(rows) - 5} more.")
        return

    # Append to metadata CSV
    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Appended {len(rows)} rows to {output_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
