"""
Packages Reddit-scraped images into a HuggingFace dataset.

Writes parquet shards to disk one at a time (low RAM),
then uploads them to the Hub via upload_folder.

Usage:
    python package_reddit_dataset.py
    python package_reddit_dataset.py --dry-run
    python package_reddit_dataset.py --shard-size 500
    python package_reddit_dataset.py --skip-build   # just upload existing shards
"""

import os
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login, HfApi

login(token=os.environ.get("HF_TOKEN"))

REDDIT_STAGING_DIR = "data/reddit_images"
METADATA_CSV = os.path.join(REDDIT_STAGING_DIR, "reddit_metadata.csv")
HUB_REPO = "ComplexDataLab/OpenFakeV2"
CONFIG_NAME = "reddit"
PARQUET_OUTPUT_DIR = "data/parquet_staging/reddit"
SHARD_SIZE = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--metadata", type=str, default=METADATA_CSV)
    parser.add_argument("--staging-dir", type=str, default=REDDIT_STAGING_DIR)
    parser.add_argument("--repo", type=str, default=HUB_REPO)
    parser.add_argument("--config", type=str, default=CONFIG_NAME)
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument("--output-dir", type=str, default=PARQUET_OUTPUT_DIR)
    parser.add_argument("--skip-build", action="store_true", help="Skip parquet building, just upload existing shards")
    return parser.parse_args()


def image_to_bytes(image_path):
    with open(image_path, "rb") as f:
        return f.read()


def build_parquet_shards(df, staging_dir, output_dir, shard_size):
    os.makedirs(output_dir, exist_ok=True)

    total = len(df)
    num_shards = (total + shard_size - 1) // shard_size
    print(f"Building {num_shards} parquet shards ({shard_size} images each)...")

    for shard_idx in range(num_shards):
        shard_path = os.path.join(output_dir, f"train-{shard_idx:05d}-of-{num_shards:05d}.parquet")

        if os.path.exists(shard_path):
            print(f"  Shard {shard_idx + 1}/{num_shards}: already exists, skipping.")
            continue

        start = shard_idx * shard_size
        end = min(start + shard_size, total)
        shard_df = df.iloc[start:end]

        image_bytes_list = []
        labels = []
        subreddits = []
        post_dates = []
        reddit_ids = []
        skipped = 0

        for _, row in shard_df.iterrows():
            image_path = os.path.join(staging_dir, row["filename"])
            if not os.path.exists(image_path):
                skipped += 1
                continue

            image_bytes_list.append({"bytes": image_to_bytes(image_path), "path": row["filename"]})
            labels.append(str(row.get("label", "") or ""))
            subreddits.append(str(row.get("subreddit", "") or ""))
            post_dates.append(str(row.get("post_date", "") or ""))
            reddit_ids.append(str(row.get("reddit_id", "") or ""))

        table = pa.table({
            "image": image_bytes_list,
            "label": labels,
            "subreddit": subreddits,
            "post_date": post_dates,
            "reddit_id": reddit_ids,
        })

        pq.write_table(table, shard_path)
        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        print(f"  Shard {shard_idx + 1}/{num_shards}: {len(image_bytes_list)} images, {size_mb:.0f}MB" +
            (f" ({skipped} missing)" if skipped else ""))

        del table, image_bytes_list, labels, subreddits, post_dates, reddit_ids

    print(f"All shards written to {output_dir}")


def main():
    args = parse_args()

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    print(f"Total rows: {len(df)}")

    for col in ["subreddit", "post_date", "reddit_id"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    print(f"  Subreddits: {df['subreddit'].nunique()}")

    if args.dry_run:
        num_shards = (len(df) + args.shard_size - 1) // args.shard_size
        print(f"\n[DRY RUN] Would build {num_shards} parquet shards to {args.output_dir}")
        print(f"Then upload to {args.repo} (config={args.config})")
        print("First 5 rows:")
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            print(f"  {row['filename']} | label={row['label']} | sub={row['subreddit']}")
        return

    if not args.skip_build:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        build_parquet_shards(df, args.staging_dir, args.output_dir, args.shard_size)
    else:
        print("Skipping parquet build, uploading existing shards...")

    print(f"\nUploading shards to {args.repo} (config={args.config})...")
    api = HfApi()
    api.upload_folder(
        repo_id=args.repo,
        repo_type="dataset",
        folder_path=args.output_dir,
        path_in_repo=f"{args.config}/train",
    )
    print("Upload complete.")

    print("Marking all rows as packaged...")
    full_df = pd.read_csv(args.metadata)
    full_df["packaged"] = True
    full_df.to_csv(args.metadata, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
