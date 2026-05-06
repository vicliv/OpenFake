#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from eval import evaluate_checkpoint

CHECKPOINT_FILENAMES = {
    "genimage_swinv2": "genimage.safetensors",
    "semitruths_swinv2": "semi_truths.safetensors",
    "of_v2_swinv2": "openfake.safetensors",
    "sofake_v2_swinv2": "sofake.safetensors",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=CHECKPOINT_FILENAMES, required=True)
    ap.add_argument("--weights_root", required=True, help="Directory containing the released SwinV2 .safetensors files.")
    ap.add_argument("--checkpoint", default=None, help="Override checkpoint path for --which.")
    ap.add_argument("--test_csv", action="append", required=True)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--write_predictions", action="store_true")
    args = ap.parse_args()
    checkpoint = Path(args.checkpoint) if args.checkpoint else Path(args.weights_root) / CHECKPOINT_FILENAMES[args.which]
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    out_dir = Path(args.out_dir or f"results/{args.which}")
    for csv in args.test_csv:
        evaluate_checkpoint(
            str(checkpoint),
            csv,
            str(out_dir / (Path(csv).stem + ".csv")),
            num_workers=args.num_workers,
            write_predictions=args.write_predictions,
        )


if __name__ == "__main__":
    main()
