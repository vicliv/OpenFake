"""
Uploads pre-built parquet shards to HuggingFace, batching multiple files
per commit to avoid the 128 commits/hour rate limit.

Resumable: tracks uploaded files in .uploaded.json.

Usage:
    python package_reddit_dataset.py                # upload all shards
    python package_reddit_dataset.py --dry-run      # preview
    python package_reddit_dataset.py --batch-size 5 # fewer files per commit if OOM
"""

import os
import glob
import json
import argparse
import time
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login, HfApi, CommitOperationAdd

login(token=os.environ.get("HF_TOKEN"))

PARQUET_OUTPUT_DIR = "data/parquet_staging/reddit"
HUB_REPO = "ComplexDataLab/OpenFakeV2"
CONFIG_NAME = "reddit"
UPLOAD_TRACKER = os.path.join(PARQUET_OUTPUT_DIR, ".uploaded.json")
BATCH_SIZE = 10  # files per commit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", type=str, default=HUB_REPO)
    parser.add_argument("--config", type=str, default=CONFIG_NAME)
    parser.add_argument("--parquet-dir", type=str, default=PARQUET_OUTPUT_DIR)
    parser.add_argument("--metadata", type=str, default="data/reddit_images/reddit_metadata.csv")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of parquet files per commit (default 10)")
    return parser.parse_args()


def load_uploaded(tracker_path):
    if os.path.exists(tracker_path):
        with open(tracker_path, "r") as f:
            return set(json.load(f))
    return set()


def save_uploaded(tracker_path, uploaded):
    with open(tracker_path, "w") as f:
        json.dump(sorted(uploaded), f)


def main():
    args = parse_args()
    api = HfApi()

    shards = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    print(f"Found {len(shards)} parquet shards in {args.parquet_dir}")

    total_size = sum(os.path.getsize(s) for s in shards)
    print(f"Total size: {total_size / (1024**3):.1f} GB")

    uploaded = load_uploaded(UPLOAD_TRACKER)
    remaining = [s for s in shards if os.path.basename(s) not in uploaded]
    print(f"Already uploaded: {len(uploaded)}, remaining: {len(remaining)}")

    if not remaining:
        print("All shards already uploaded.")
    elif args.dry_run:
        num_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
        print(f"\n[DRY RUN] Would upload {len(remaining)} files in {num_batches} commits")
        print(f"  Batch size: {args.batch_size} files per commit")
        print(f"  Destination: {args.repo}/{args.config}/train/")
        for s in remaining[:5]:
            size_mb = os.path.getsize(s) / (1024**2)
            print(f"  {os.path.basename(s)}: {size_mb:.0f} MB")
        if len(remaining) > 5:
            print(f"  ... and {len(remaining) - 5} more")
        return
    else:
        # Upload in batches
        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start:batch_start + args.batch_size]
            batch_num = batch_start // args.batch_size + 1
            total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size

            batch_size_mb = sum(os.path.getsize(s) for s in batch) / (1024**2)
            filenames = [os.path.basename(s) for s in batch]
            print(f"\n  Batch {batch_num}/{total_batches}: {len(batch)} files ({batch_size_mb:.0f} MB)", flush=True)

            operations = [
                CommitOperationAdd(
                    path_in_repo=f"{args.config}/train/{os.path.basename(shard_path)}",
                    path_or_fileobj=shard_path,
                )
                for shard_path in batch
            ]

            api.create_commit(
                repo_id=args.repo,
                repo_type="dataset",
                operations=operations,
                commit_message=f"Add {args.config} shards: {filenames[0]} to {filenames[-1]}",
            )

            for s in batch:
                uploaded.add(os.path.basename(s))
            save_uploaded(UPLOAD_TRACKER, uploaded)
            print(f"    Committed {len(batch)} files.", flush=True)

    # Mark all rows as packaged
    metadata_csv = args.metadata
    if os.path.exists(metadata_csv):
        print("\nMarking all rows as packaged...")
        import pandas as pd
        df = pd.read_csv(metadata_csv)
        df["packaged"] = True
        df.to_csv(metadata_csv, index=False)
        print(f"Marked {len(df)} rows as packaged.")

    print("Done.")


if __name__ == "__main__":
    main()
