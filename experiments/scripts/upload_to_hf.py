import os

os.environ["HF_DATASETS_CACHE"] = os.environ.get("SLURM_TMPDIR", "/tmp")
os.environ["HF_HOME"] = os.environ.get("SLURM_TMPDIR", "/tmp")

import argparse
import time
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Image, Value
from huggingface_hub import whoami
from huggingface_hub.utils import HfHubHTTPError

REPO_ID = "ComplexDataLab/OpenFake"
SPLITS_DIR = Path(__file__).resolve().parents[2] / "OpenFake" / "splits"

features = Features({
    "image": Image(),
    "prompt": Value("string"),
    "label": Value("string"),
    "model": Value("string"),
    "type": Value("string"),
    "release_date": Value("string"),
})


def build_split(csv_path: Path) -> Dataset:
    df = pd.read_csv(csv_path)

    if df["file_name"].iloc[0].startswith("/"):
        df["image"] = df["file_name"]
    else:
        src_root = Path(os.environ["SCRATCH"]) / "OpenFake"
        print(f"  WARNING: file_name is not absolute in {csv_path.name}; prepending {src_root}")
        df["image"] = df["file_name"].apply(lambda f: str(src_root / f))

    for col in ["prompt", "model", "type", "release_date"]:
        if col not in df.columns:
            df[col] = None
        else:
            df[col] = df[col].astype("object").where(df[col].notna(), None)

    df = df[["image", "prompt", "label", "model", "type", "release_date"]]
    ds = Dataset.from_pandas(df, features=features, preserve_index=False)
    return ds.shuffle(seed=42)


def push_with_retry(ds: Dataset, split_name: str, config_name: str, set_default: bool, dry_run: bool):
    if dry_run:
        return

    for attempt in range(5):
        try:
            ds.push_to_hub(
                REPO_ID,
                config_name=config_name,
                split=split_name,
                max_shard_size="5GB",
                set_default=set_default,
            )
            break
        except (HfHubHTTPError, ConnectionError) as e:
            wait = 60 * (2 ** attempt)
            print(f"  upload failed (attempt {attempt + 1}/5): {e}; retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"push_to_hub failed after 5 attempts for {config_name}/{split_name}")


def print_stats(ds: Dataset, split_name: str):
    labels = ds["label"]
    n = len(labels)
    n_fake = sum(1 for l in labels if l == "fake")
    n_real = n - n_fake
    print(f"  {split_name}: {n:,} rows  fake={n_fake:,} ({100*n_fake/n:.1f}%)  real={n_real:,} ({100*n_real/n:.1f}%)")


def run_core(dry_run: bool):
    splits = {
        "train":      SPLITS_DIR / "of_train_v2.csv",
        "validation": SPLITS_DIR / "of_test_indist_v2.csv",
        "test":       SPLITS_DIR / "of_test_ood_models_v2.csv",
    }

    for i, (split_name, csv_path) in enumerate(splits.items()):
        print(f"\nBuilding {split_name} from {csv_path.name}")
        ds = build_split(csv_path)
        print_stats(ds, split_name)
        if not dry_run:
            print(f"Pushing {split_name} → config=core")
        push_with_retry(ds, split_name, "core", set_default=(i == 0), dry_run=dry_run)
        if not dry_run:
            print(f"  done.")


def run_reddit(dry_run: bool):
    csv_path = SPLITS_DIR / "reddit.csv"
    print(f"\nBuilding reddit/test from {csv_path.name}")
    ds = build_split(csv_path)
    print_stats(ds, "test")
    if not dry_run:
        print("Pushing test → config=reddit")
    push_with_retry(ds, "test", "reddit", set_default=False, dry_run=dry_run)
    if not dry_run:
        print("  done.")


def verify_preconditions():
    from huggingface_hub import list_repo_refs
    tags = [t.name for t in list_repo_refs(REPO_ID, repo_type="dataset").tags]
    if "v1.0" not in tags:
        raise RuntimeError("v1.0 tag missing on ComplexDataLab/OpenFake — do not proceed until tagged")
    print("v1.0 tag verified.")

    for fname in ["of_train_v2.csv", "of_test_indist_v2.csv", "of_test_ood_models_v2.csv", "reddit.csv"]:
        p = SPLITS_DIR / fname
        if not p.exists():
            raise FileNotFoundError(f"Split CSV not found: {p}")
    print("All split CSVs present.")

    import random
    for fname in ["of_train_v2.csv", "of_test_indist_v2.csv", "of_test_ood_models_v2.csv", "reddit.csv"]:
        df = pd.read_csv(SPLITS_DIR / fname)
        sample = df.sample(min(100, len(df)), random_state=42)
        missing = [f for f in sample["file_name"] if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f"{fname}: {len(missing)} paths do not exist, e.g. {missing[0]}")
    print("Path spot-checks passed.")

    me = whoami()
    print(f"HF auth OK: {me['name']}")


def main():
    parser = argparse.ArgumentParser(description="Upload OpenFake v2 to HuggingFace")
    parser.add_argument("--config", choices=["core", "reddit"], required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Build datasets and print sizes without pushing")
    parser.add_argument("--skip-prechecks", action="store_true",
                        help="Skip precondition verification (for reruns after initial check)")
    args = parser.parse_args()

    if not args.skip_prechecks:
        print("=== Verifying preconditions ===")
        verify_preconditions()

    if args.dry_run:
        print("\n=== DRY RUN (no data will be pushed) ===")
    else:
        print(f"\n=== Uploading config={args.config} to {REPO_ID} ===")

    if args.config == "core":
        run_core(dry_run=args.dry_run)
    else:
        run_reddit(dry_run=args.dry_run)

    if args.dry_run:
        print("\nDry run complete. Run without --dry-run to push.")
    else:
        print("\nUpload complete.")


if __name__ == "__main__":
    main()
