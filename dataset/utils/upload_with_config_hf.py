"""
Packages HF-generated images into a HuggingFace dataset.

Reads metadata.csv from the staging directory, builds a HF Dataset,
pushes to the Hub, then marks all rows as packaged.

Usage:
    python package_opensource_dataset.py
    python package_opensource_dataset.py --dry-run
"""

import os
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, Image, Features, Value
from dotenv import load_dotenv

load_dotenv()
from huggingface_hub import login
login(token=os.environ.get("HF_TOKEN"))

STAGING_DIR = "data/staging_images"
METADATA_CSV = os.path.join(STAGING_DIR, "metadata.csv")
HUB_REPO = "ComplexDataLab/OpenFakeV2"
CONFIG_NAME = "opensource"

features = Features({
    "image": Image(),
    "prompt": Value("string"),
    "label": Value("string"),
    "model": Value("string"),
    "type": Value("string"),
    "release_date": Value("string"),
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without pushing to Hub.")
    parser.add_argument("--metadata", type=str, default=METADATA_CSV)
    parser.add_argument("--staging-dir", type=str, default=STAGING_DIR)
    parser.add_argument("--repo", type=str, default=HUB_REPO)
    parser.add_argument("--config", type=str, default=CONFIG_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    staging_dir = Path(args.staging_dir)

    print(f"Loading metadata from {args.metadata}...")
    df = pd.read_csv(args.metadata)
    print(f"Total rows: {len(df)}")

    # Resolve image paths and verify they exist
    df["image_path"] = df["filename"].apply(lambda f: str(staging_dir / f))
    missing = df[~df["image_path"].apply(os.path.exists)]
    if len(missing) > 0:
        print(f"WARNING: {len(missing)} images missing from disk. Skipping.")
        df = df[df["image_path"].apply(os.path.exists)].copy()
        print(f"Remaining: {len(df)}")

    if len(df) == 0:
        print("No valid images. Exiting.")
        return

    # Fill missing optional fields
    for col in ["prompt", "model", "type", "release_date"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")

    # Build dataset dataframe
    ds_df = df[["image_path", "prompt", "label", "model", "type", "release_date"]].copy()
    ds_df = ds_df.rename(columns={"image_path": "image"})

    print(f"\nBuilding dataset with {len(ds_df)} rows...")
    ds = Dataset.from_pandas(ds_df, features=features, preserve_index=False)
    ds = ds.shuffle(seed=42)

    # Summary
    print(f"  Models: {ds_df['model'].nunique()}")
    print(f"  Labels: {ds_df['label'].value_counts().to_dict()}")
    print(f"  Types: {ds_df['type'].value_counts().to_dict()}")

    if args.dry_run:
        print(f"\n[DRY RUN] Would push {len(ds)} rows to {args.repo} (config={args.config}, split=train)")
        for i in range(min(5, len(ds_df))):
            row = ds_df.iloc[i]
            print(f"  {os.path.basename(row['image'])} | model={row['model']} | type={row['type']}")
        return

    print(f"\nPushing to {args.repo} (config={args.config}, split=train)...")
    ds.push_to_hub(
        args.repo,
        split="train",
        max_shard_size="5GB",
        set_default=True,
        config_name=args.config,
    )
    print("Push complete.")

    # Mark all rows as packaged
    print("Marking all rows as packaged...")
    full_df = pd.read_csv(args.metadata)
    full_df["packaged"] = True
    full_df.to_csv(args.metadata, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
