# OpenFake

Code for building the OpenFake dataset and reproducing the detector experiments from the paper.

The repository has two independent parts:

- `dataset/`: data collection and packaging code for the OpenFake dataset.
- `experiments/`: training, evaluation, split creation, and paper-table utilities.

Large artifacts are intentionally not versioned. Keep local images, split CSVs, results, paper tables, and model weights outside git or under ignored directories such as `data/`, `splits/`, `results/`, `paper_tables/`, and `model_weights/`.

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/) with `pyproject.toml`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Run commands through the project environment:

```bash
uv run python experiments/train.py --help
uv run python experiments/scripts/build_of_splits.py --help
```

For GPU training, install the PyTorch build that matches your CUDA environment if the default resolver does not choose the correct wheel for your system.

## Data Setup

Create local working directories:

```bash
mkdir -p data/openfake_hf data/hf_cache data/competitor_eval splits results model_weights
```

OpenFake is downloaded from Hugging Face by the split builder:

```bash
uv run python experiments/scripts/build_of_splits.py \
  --openfake-image-dir data/openfake_hf \
  --hf-cache-dir data/hf_cache \
  --docci-dir /path/to/docci/images \
  --imagenet-dir /path/to/imagenet/train \
  --out-dir splits \
  --results-dir results
```

The script reads `ComplexDataLab/OpenFake` configs `opensource`, `reddit`, and `inpainting`, materializes image payloads into `--openfake-image-dir`, and writes:

- `splits/of_train_v2.csv`
- `splits/of_test_indist_v2.csv`
- `splits/of_test_ood_models_v2.csv`

External evaluation CSVs are created with:

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

Dataset sources:

- OpenFake: Hugging Face dataset `ComplexDataLab/OpenFake`.
- Community Forensics Eval: `OwensLab/CommunityForensics-Eval`, downloaded by `download_eval_sets.py`.
- So-Fake-OOD: `saberzl/So-Fake-OOD`, downloaded by `download_eval_sets.py`.
- Semi-Truths Eval: `semi-truths/Semi-Truths-Evalset`, downloaded by `download_eval_sets.py`.
- GenImage: download from the official GenImage release, then pass the unpacked root containing `metadata.csv` as `--genimage-root`.
- DOCCI: download `google/docci` and pass the image directory as `--docci-dir`.
- ImageNet: download ILSVRC2012 train images through the official ImageNet access flow and pass the unpacked `train/` directory as `--imagenet-dir`.

## Continuously Updated Dataset Pipeline

OpenFake was designed as a continuously updated dataset rather than a one-time static scrape. The generation and Reddit scripts append new images and metadata to local staging directories, while registry/metadata files keep enough state to resume later runs without reprocessing completed work.

There are three update streams:

- Text-to-image generation: `dataset/huggingface_pipeline.py` scans Hugging Face Diffusers models, filters eligible models, downloads weights, generates images from the prompt CSV, appends rows to `data/staging_images/metadata.csv`, and records model status in `data/model_registry.json`.
- Inpainting generation: `dataset/inpaint_pipeline.py` uses Open Images masks and generated prompts, downloads only the needed source photos for each run, writes inpainted images to `data/staging_inpaint_images/`, and records status in `data/inpaint_model_registry.json`.
- Reddit collection: `dataset/reddit_scraper.py` reads the existing `reddit_metadata.csv`, resumes each subreddit from the latest scraped post date, downloads new image posts and video frames, and appends rows to `data/reddit_images/reddit_metadata.csv`.

A typical repeated update looks like:

```bash
# Add new synthetic images from eligible Hugging Face text-to-image models.
uv run python dataset/huggingface_pipeline.py \
  --staging-dir data/staging_images \
  --registry-file data/model_registry.json \
  --slurm-log-dir data/slurm_logs \
  --txt2img-script /path/to/scheduler-wrapper

# Add new Reddit images since the last recorded post per subreddit.
uv run python dataset/reddit_scraper.py \
  --creds-csv data/creds/creds.csv \
  --staging-dir data/reddit_images \
  --metadata-csv data/reddit_images/reddit_metadata.csv
```

The text-to-image registry has `COMPLETED`, `MODEL_FAULT`, and `INFRASTRUCTURE_FAULT` states. Future runs skip completed and model-fault entries, and retry infrastructure faults. The Reddit scraper is date-resumable per subreddit; use `--days-ago N` only when you want to force a fixed lookback instead of resume mode.

After collecting more rows, package and upload the updated configs with the dataset utilities in `dataset/README.md`. Use `--dry-run` first to inspect what would be pushed.

## Model Weights

Paper weights are not tracked in git. Put them under `model_weights/`.

The OpenFake, So-Fake, Semi-Truths, GenImage, DRCT, DeepFakeBench, and CLIP baseline weights used by the scripts are expected under the structure shown by `model_weights/` in the local working copy. The shared Google Drive folder for released weights is:

https://drive.google.com/drive/folders/1xdwktgvvc9uSjVee5ZERiYlEUV3qBbwv?usp=share_link

The C-F checkpoint is downloaded from Hugging Face by `experiments/scripts/eval_cf.py`.

## Reproduction

See `experiments/README.md` for training and evaluation commands. See `dataset/README.md` for dataset creation and packaging commands.

## License

This work — including the code in this repository and the released model
weights — is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are free to share and adapt the material for non-commercial purposes,
provided you give appropriate credit. Commercial use is not permitted without
separate permission.

Third-party components retain their original licenses

The released weights are also distributed under CC BY-NC 4.0 (see the License section below).