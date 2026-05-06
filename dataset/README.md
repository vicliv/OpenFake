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

## Continuous Update Model

The dataset scripts are intended to be run repeatedly. Each pipeline writes images into a staging directory and appends metadata rows to a CSV. State files prevent duplicated work:

- `data/model_registry.json`: text-to-image model status for Hugging Face generation.
- `data/inpaint_model_registry.json`: inpainting model status.
- `data/reddit_images/reddit_metadata.csv`: downloaded Reddit filenames and post dates, used to resume by subreddit.
- `packaged` metadata columns: used by packaging utilities to mark rows already included in a Hub upload.

This means a normal update cycle is:

1. Run one or more collectors/generators to append new rows.
2. Optionally filter or inspect staged images.
3. Package the updated rows into a Hugging Face dataset config.
4. Re-run later with the same registry and metadata paths to continue from the previous state.

## Text-to-Image Pipeline

The Hugging Face pipeline scans Diffusers models, downloads eligible weights, and submits generation jobs to the configured scheduler script. It is resumable through `--registry-file`: completed models are skipped, incompatible models are marked as model faults, and temporary infrastructure failures are retried on the next run.

For a continuous update across eligible Hugging Face text-to-image models:

```bash
uv run python dataset/huggingface_pipeline.py \
  --staging-dir data/staging_images \
  --registry-file data/model_registry.json \
  --slurm-log-dir data/slurm_logs \
  --txt2img-script /path/to/scheduler-wrapper
```

Outputs:

- images in `data/staging_images/`
- metadata in `data/staging_images/metadata.csv`
- model status in `data/model_registry.json`

The metadata rows include filename, prompt, label, model ID, model type, release date, and packaging status. Re-running the command with the same registry continues the scan without regenerating completed models.

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

Direct worker calls do not update model registry status; use the pipeline entry point for the continuous update workflow.

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

Run inpainting for one or more models:

```bash
uv run python dataset/inpaint_pipeline.py \
  --model black-forest-labs/FLUX.1-dev \
  --staging-dir data/staging_inpaint_images \
  --master-prompts-csv data/open_images/master_prompts_300k.csv \
  --masks-dir data/open_images/masks \
  --inpaint-tmp-root data/inpaint_tmp \
  --registry-file data/inpaint_model_registry.json
```

For continuous updates, keep the same `--registry-file`, `--master-prompts-csv`, and `--masks-dir`. The pipeline samples valid mask/image rows, writes a temporary per-model manifest, appends generated images to `data/staging_inpaint_images/metadata.csv`, and skips models already marked completed or model-faulted.

## Reddit Pipeline

Create a Reddit credentials CSV with columns:

```text
datatype,client_id,client_secret,useragent
```

Then run in resume mode:

```bash
uv run python dataset/reddit_scraper.py \
  --creds-csv data/creds/creds.csv \
  --staging-dir data/reddit_images \
  --metadata-csv data/reddit_images/reddit_metadata.csv \
  --skip-nsfw-filter
```

Resume mode reads `reddit_metadata.csv`, finds the most recent `post_date` per subreddit, and only downloads newer posts. Image posts are saved directly; supported Reddit videos are converted into evenly spaced frame images. Every saved image/frame gets a metadata row with its label, subreddit, post date, Reddit ID, and packaging status.

Use a fixed lookback when bootstrapping or intentionally backfilling:

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
