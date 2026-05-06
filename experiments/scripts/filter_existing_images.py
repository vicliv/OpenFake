#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


DEFAULT_CSV = Path(__file__).resolve().parents[1] / "splits" / "semitruths_eval.csv"


def filter_existing_images(csv_path: Path, out_path: Path | None, dry_run: bool) -> None:
    df = pd.read_csv(csv_path, dtype=str, low_memory=False).fillna("")
    if "file_name" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a 'file_name' column")

    exists = df["file_name"].map(lambda p: Path(p).exists())
    kept = int(exists.sum())
    removed = int((~exists).sum())

    print(f"Input: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Keeping: {kept}")
    print(f"Removing missing files: {removed}")

    if dry_run:
        return

    target = out_path or csv_path
    if target == csv_path:
        backup = csv_path.with_suffix(csv_path.suffix + ".bak")
        shutil.copy2(csv_path, backup)
        print(f"Backup written: {backup}")

    target.parent.mkdir(parents=True, exist_ok=True)
    df.loc[exists].to_csv(target, index=False)
    print(f"Filtered CSV written: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove CSV rows whose image path in the file_name column does not exist."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV to filter.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV. If omitted, overwrite --csv after writing a .bak backup.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report counts; do not write files.")
    args = parser.parse_args()

    filter_existing_images(args.csv, args.out, args.dry_run)


if __name__ == "__main__":
    main()
