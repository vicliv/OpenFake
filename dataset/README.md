# OpenFake Dataset Code

This directory contains the data-collection and packaging scripts used to build the OpenFake dataset. The released dataset is hosted on Hugging Face as `ComplexDataLab/OpenFake`.

Use the repository-level `uv` environment:

```bash
uv sync
uv run python dataset/huggingface_pipeline.py --help
```

All data paths are command-line arguments or portable relative defaults. Do not edit scripts to point at machine-specific storage; pass paths with flags.

## Directory Layout

Recommended local layout:

```text
data/
  staging_images/
  staging_inpaint_images/
  reddit_images/
  open_images/
    masks/
    prompt_shards/
    master_prompts_300k.csv
  prompts/
    unused_prompts2.csv
  creds/
    creds.csv
  models/
```

These directories are ignored by git.

## Text-to-Image Pipeline

The Hugging Face pipeline scans Diffusers models, downloads eligible weights, and submits generation jobs to the configured scheduler script.

```bash
uv run python dataset/huggingface_pipeline.py \
  --staging-dir data/staging_images \
  --registry-file data/model_registry.json \
  --slurm-log-dir data/slurm_logs \
  --txt2img-script /path/to/scheduler-wrapper
```

For a single model:

```bash
uv run python dataset/huggingface_pipeline.py \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --staging-dir data/staging_images \
  --registry-file data/model_registry.json
```

The compute worker can also be called directly:

```bash
uv run python dataset/generate_batch_samples.py \
  stabilityai/stable-diffusion-xl-base-1.0 \
  10000 \
  Base \
  None \
  2023-07-26 \
  --staging-dir data/staging_images \
  --prompts-csv data/prompts/unused_prompts2.csv
```

## Inpainting Pipeline

Prepare Open Images metadata and prompt shards first, then download required masks:

```bash
uv run python dataset/generate_inpaint_prompts.py \
  --metadata-dir data/open_images/zoo/open-images-v7/train/metadata \
  --labels-dir data/open_images/zoo/open-images-v7/train/labels \
  --output-dir data/open_images/prompt_shards \
  --final-output data/open_images/master_prompts_300k.csv

uv run python dataset/utils/download-inpainting-masks.py \
  --shard-dir data/open_images/prompt_shards \
  --masks-dir data/open_images/masks \
  --zips-dir data/open_images/tmp_zips
```

Run inpainting:

```bash
uv run python dataset/inpaint_pipeline.py \
  --model black-forest-labs/FLUX.1-dev \
  --staging-dir data/staging_inpaint_images \
  --master-prompts-csv data/open_images/master_prompts_300k.csv \
  --masks-dir data/open_images/masks \
  --inpaint-tmp-root data/inpaint_tmp \
  --registry-file data/inpaint_model_registry.json
```

## Reddit Pipeline

Create a Reddit credentials CSV with columns:

```text
datatype,client_id,client_secret,useragent
```

Then run:

```bash
uv run python dataset/reddit_scraper.py \
  --creds-csv data/creds/creds.csv \
  --staging-dir data/reddit_images \
  --metadata-csv data/reddit_images/reddit_metadata.csv \
  --days-ago 30 \
  --skip-nsfw-filter
```

If you have a scheduler script for NSFW filtering, omit `--skip-nsfw-filter` and pass `--nsfw-script`.

Manual NSFW filtering:

```bash
uv run python dataset/filter_nsfw.py \
  data/reddit_images \
  --model-path data/models/nsfw_filtering_model \
  --threshold 0.7
```

## Packaging For Hugging Face

Package generated text-to-image rows:

```bash
uv run python dataset/utils/upload_with_config_hf.py \
  --metadata data/staging_images/metadata.csv \
  --staging-dir data/staging_images \
  --repo ComplexDataLab/OpenFake \
  --config opensource \
  --dry-run
```

Package Reddit rows:

```bash
uv run python dataset/utils/upload_with_config_reddit.py \
  --metadata data/reddit_images/reddit_metadata.csv \
  --staging-dir data/reddit_images \
  --output-dir data/parquet_staging/reddit \
  --repo ComplexDataLab/OpenFake \
  --config reddit \
  --dry-run
```

Remove `--dry-run` only when the target Hub repository and token are configured.
