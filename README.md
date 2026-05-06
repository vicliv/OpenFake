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

## Model Weights

Paper weights are not tracked in git. Put them under `model_weights/`.

The OpenFake, So-Fake, Semi-Truths, GenImage, DRCT, DeepFakeBench, and CLIP baseline weights used by the scripts are expected under the structure shown by `model_weights/` in the local working copy. The shared Google Drive folder for released weights is:

https://drive.google.com/drive/folders/1xdwktgvvc9uSjVee5ZERiYlEUV3qBbwv?usp=share_link

The C-F checkpoint is downloaded from Hugging Face by `experiments/scripts/eval_cf.py`.

## Reproduction

See `experiments/README.md` for training and evaluation commands. See `dataset/README.md` for dataset creation and packaging commands.
