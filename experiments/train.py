"""Training entry point for OpenFake deepfake detection.

Usage (single GPU):
    python train.py --config configs/default.yaml

Usage (multi-GPU via accelerate):
    accelerate launch --num_processes=4 train.py --config configs/default.yaml

Multiple configs are merged left-to-right (later files override earlier ones):
    accelerate launch --num_processes=4 train.py \\
        --config configs/default.yaml \\
        --config configs/vit.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime
import torch.multiprocessing

# SLURM cgroups count /dev/shm against the memory limit; with 4 processes ×
# 6 workers × prefetch_factor=2 the shared-memory ring buffers OOM the job.
# file_system uses TMPDIR instead — no /dev/shm pressure. Point it at
# SLURM_TMPDIR (local NVMe SSD) when available, otherwise fall back to /tmp.
os.environ.setdefault("TMPDIR", os.environ.get("SLURM_TMPDIR", "/tmp"))
torch.multiprocessing.set_sharing_strategy("file_system")

from transformers import set_seed

from config import load_config
from augmentations.transforms import build_train_transform, build_eval_transform
from data.collator import OpenFakeCollator
from data.splits import build_datasets
from models.build import build_model
from training.metrics import compute_metrics, compute_per_model_metrics
from training.trainer import build_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an OpenFake classifier")
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        metavar="PATH",
        help="YAML config file (repeatable; later files override earlier ones)",
    )
    parser.add_argument("--openfake-root", default=None, help="Root for directory-based metadata splits.")
    parser.add_argument("--output-dir", default=None, help="Checkpoint/output directory for this run.")
    parser.add_argument("--model-cache-dir", default=None, help="Model cache directory.")
    parser.add_argument("--results-dir", default=None, help="Directory for per-split evaluation CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config:
        default_cfg = os.path.join(os.path.dirname(__file__), "configs", "default.yaml")
        args.config = [default_cfg]

    cfg = load_config(*args.config)
    if args.openfake_root:
        cfg.data.openfake_root = args.openfake_root
    if args.output_dir:
        cfg.training.output_dir = args.output_dir
    if args.model_cache_dir:
        cfg.model.cache_dir = args.model_cache_dir

    if cfg.training.output_dir is None:
        run_name = cfg.training.wandb_run_name or cfg.model.name
        cfg.training.output_dir = os.path.abspath(os.path.join("runs", run_name))
        print(f"output_dir not set in config — using: {cfg.training.output_dir}")

    os.makedirs(cfg.training.output_dir, exist_ok=True)
    results_root = os.path.abspath(args.results_dir or os.path.join(os.path.dirname(__file__), "..", "results"))
    run_name = cfg.training.wandb_run_name or os.path.basename(cfg.training.output_dir)
    run_results_dir = os.path.join(results_root, run_name)
    os.makedirs(run_results_dir, exist_ok=True)
    os.environ["WANDB_PROJECT"] = cfg.training.wandb_project
    os.environ.setdefault("WANDB_RESUME", "allow")

    set_seed(42)

    # ---- Model ----
    processor, model = build_model(cfg.model)

    # ---- Transforms ----
    train_transform = build_train_transform(cfg.augment)
    eval_transform = build_eval_transform(cfg.augment)

    # ---- Datasets ----
    train_ds, val_ds, test_ds_map = build_datasets(cfg, train_transform, eval_transform)

    # ---- Trainer ----
    collator = OpenFakeCollator(processor)
    trainer, resume_ckpt = build_trainer(cfg, model, train_ds, val_ds, compute_metrics, collator)

    # ---- Train ----
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # ---- Evaluate on each test split ----
    for split_name, test_ds in test_ds_map.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on test split: {split_name}")
        results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix=f"test_{split_name}")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        pred = trainer.predict(test_ds)
        model_names = [s[2] if len(s) > 2 else "" for s in test_ds.samples]
        df = compute_per_model_metrics(pred.predictions, pred.label_ids, model_names)
        out_csv = os.path.join(run_results_dir, f"{split_name}.csv")
        df.to_csv(out_csv, index=False)
        with open(out_csv + ".json", "w") as f:
            json.dump(
                {
                    "run_name": run_name,
                    "test_set": split_name,
                    "rows": len(test_ds),
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "git_sha": None,
                    "config": args.config,
                },
                f,
                indent=2,
            )
        print(f"  wrote per-model metrics: {out_csv}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
