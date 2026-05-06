"""
Downloads ONLY the masks needed for the inpainting pipeline.
Reads the prompt shard CSVs (or merged master CSV) to determine exactly which masks to extract.

Images are NOT downloaded here — the inpainting pipeline handles those temporarily per model run.

Each mask zip (~700MB-1GB) is downloaded, only the needed PNGs are extracted, then the zip is deleted.
At most 1 zip is on disk at a time.

Usage:
    python download_masks.py                     # extract masks for all prompt shards
    python download_masks.py --max-masks 1000    # limit for testing
"""

import argparse
import os
import glob
import csv
import zipfile
import subprocess
from collections import defaultdict

SHARD_DIR = "data/open_images/prompt_shards"
MASKS_DIR = "data/open_images/masks"
ZIPS_DIR = "data/open_images/tmp_zips"

MASK_ZIP_BASE_URL = "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-masks", type=int, default=None, help="Limit total masks for testing")
    parser.add_argument("--shard-dir", default=SHARD_DIR)
    parser.add_argument("--masks-dir", default=MASKS_DIR)
    parser.add_argument("--zips-dir", default=ZIPS_DIR)
    return parser.parse_args()


def load_all_shards():
    """Load all mask paths from prompt shard CSVs."""
    mask_paths = set()
    files = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.csv")))

    if not files:
        print(f"No shard files found in {SHARD_DIR}", flush=True)
        return mask_paths

    for f in files:
        with open(f, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                mask_paths.add(row["MaskPath"])
        print(f"Loaded {os.path.basename(f)}", flush=True)

    print(f"Total unique masks needed: {len(mask_paths)}", flush=True)
    return mask_paths


def download_masks(all_masks, max_masks=None):
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(ZIPS_DIR, exist_ok=True)

    # Filter out already extracted masks
    needed = set()
    already_have = 0
    for mask_path in all_masks:
        if os.path.exists(os.path.join(MASKS_DIR, mask_path)):
            already_have += 1
        else:
            needed.add(mask_path)

    if max_masks:
        needed = set(list(needed)[:max_masks])

    print(f"\n{already_have} masks already on disk, {len(needed)} to extract.", flush=True)

    if not needed:
        print("Nothing to do.", flush=True)
        return

    # Group by zip shard (first hex char)
    by_shard = defaultdict(set)
    for mask_path in needed:
        shard_char = mask_path[0].lower()
        by_shard[shard_char].add(mask_path)

    print(f"Need masks from {len(by_shard)} zip shards: {sorted(by_shard.keys())}", flush=True)

    total_extracted = 0
    for shard_char in sorted(by_shard.keys()):
        shard_masks = by_shard[shard_char]
        zip_url = f"{MASK_ZIP_BASE_URL}{shard_char}.zip"
        zip_path = os.path.join(ZIPS_DIR, f"train-masks-{shard_char}.zip")

        print(f"\n  Shard '{shard_char}': need {len(shard_masks)} masks", flush=True)

        # Download zip
        if not os.path.exists(zip_path):
            print(f"  Downloading {zip_url}...", flush=True)
            result = subprocess.run(
                ["wget", "-q", "-O", zip_path, zip_url],
                capture_output=True, text=True, timeout=1200
            )
            if result.returncode != 0 or not os.path.exists(zip_path):
                print(f"  FAILED: {result.stderr}", flush=True)
                continue
            size_mb = os.path.getsize(zip_path) / 1e6
            print(f"  Downloaded ({size_mb:.0f} MB)", flush=True)
        else:
            print(f"  Zip already cached.", flush=True)

        # Extract only needed masks
        extracted = 0
        not_found = 0
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zip_contents = set(zf.namelist())
                for mask_name in shard_masks:
                    if mask_name in zip_contents:
                        zf.extract(mask_name, MASKS_DIR)
                        extracted += 1
                    else:
                        not_found += 1
        except zipfile.BadZipFile:
            print(f"  ERROR: corrupt zip. Deleting and skipping.", flush=True)
            os.remove(zip_path)
            continue

        total_extracted += extracted
        print(f"  Extracted {extracted}, not found {not_found}.", flush=True)

        # Delete zip immediately to free space
        os.remove(zip_path)
        print(f"  Deleted zip.", flush=True)

    # Clean up
    if os.path.exists(ZIPS_DIR) and not os.listdir(ZIPS_DIR):
        os.rmdir(ZIPS_DIR)

    print(f"\nDone. Extracted {total_extracted} masks total to {MASKS_DIR}", flush=True)


def main():
    args = parse_args()
    global SHARD_DIR, MASKS_DIR, ZIPS_DIR
    SHARD_DIR = args.shard_dir
    MASKS_DIR = args.masks_dir
    ZIPS_DIR = args.zips_dir
    all_masks = load_all_shards()
    download_masks(all_masks, args.max_masks)

    # Final count
    mask_count = len([f for f in os.listdir(MASKS_DIR) if f.endswith(".png")])
    print(f"\nMasks on disk: {mask_count}", flush=True)


if __name__ == "__main__":
    main()
