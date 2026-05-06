# OpenFake Experiments

This directory contains code for creating CSV splits, training SwinV2 detectors, evaluating baselines, and generating paper tables.

Run all commands from the repository root with `uv run`.

## Prepare Splits

Build OpenFake train and evaluation CSVs directly from Hugging Face:

```bash
uv run python experiments/scripts/build_of_splits.py \
  --openfake-image-dir data/openfake_hf \
  --hf-cache-dir data/hf_cache \
  --docci-dir /path/to/docci/images \
  --imagenet-dir /path/to/imagenet/train \
  --out-dir splits \
  --results-dir results
```

Download and normalize external eval sets:

```bash
uv run python experiments/scripts/download_eval_sets.py \
  --root data/competitor_eval \
  --datasets cf sofake semitruths

uv run python experiments/scripts/build_competitor_metadata.py \
  --competitor-root data/competitor_eval \
  --genimage-root /path/to/GenImage \
  --out-dir splits \
  --results-dir results \
  --datasets all
```

Expected CSV schema:

```text
file_name,prompt,label,model,type,release_date
```

`file_name` must resolve to an image on disk. Labels are `real` or `fake`.

## Train

Train OpenFake SwinV2:

```bash
uv run python experiments/train.py \
  --config experiments/configs/default.yaml \
  --config experiments/configs/of_v2.yaml \
  --output-dir runs/of_v2 \
  --results-dir results
```

With multiple GPUs:

```bash
uv run accelerate launch --num_processes 4 experiments/train.py \
  --config experiments/configs/default.yaml \
  --config experiments/configs/of_v2.yaml \
  --output-dir runs/of_v2 \
  --results-dir results
```

The config files use repo-relative split paths such as `splits/of_train_v2.csv`. Override `--openfake-root` only when using directory-style metadata splits instead of CSV split files.

## Evaluate OpenFake-Style Checkpoints

Put local weights in `model_weights/`, then run:

```bash
uv run python experiments/scripts/eval_existing_swinv2.py \
  --which of_v2_swinv2 \
  --weights_root model_weights \
  --test_csv splits/of_test_indist_v2.csv \
  --test_csv splits/of_test_ood_models_v2.csv \
  --out_dir results/of_v2
```

Available `--which` values:

- `of_v2_swinv2`
- `sofake_v2_swinv2`
- `genimage_swinv2`
- `semitruths_swinv2`

## Baselines

C-F downloads its checkpoint from Hugging Face:

```bash
uv run python experiments/scripts/eval_cf.py \
  --test_csv splits/of_test_indist_v2.csv \
  --test_csv splits/of_test_ood_models_v2.csv \
  --out_dir results/cf_pretrained
```

DRCT, DeepFakeBench, and CLIP require their original reference code plus local weights:

```bash
uv run python experiments/scripts/baselines/eval_drct.py \
  --baseline_dir /path/to/DRCT \
  --weights_root model_weights \
  --test_csv splits/of_test_ood_models_v2.csv

uv run python experiments/scripts/baselines/eval_deepfakebench.py \
  --baseline_dir /path/to/DeepFakeBench \
  --weights_root model_weights \
  --test_csv splits/of_test_ood_models_v2.csv

uv run python experiments/scripts/baselines/eval_clip_baseline.py \
  --baseline_dir /path/to/CLIP_baseline \
  --weights_root model_weights \
  --test_csv splits/of_test_ood_models_v2.csv
```

## Paper Tables

After all evaluation CSVs are present under `results/`:

```bash
uv run python experiments/scripts/make_paper_tables.py \
  --results-dir results \
  --splits-dir splits \
  --output results/paper_tables.tex \
  --figure-dir results/paper_tables
```
