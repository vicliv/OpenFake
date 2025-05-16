# OpenFake: Deepfake Detection with SwinV2

This repository contains training and evaluation scripts for fine-tuning a SwinV2 transformer on the OpenFake dataset.

## Installation

Before running the code, install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

To train the model on the OpenFake dataset with streaming and light degradation of synthetic images:

```bash
python train.py \
    --output_dir $SCRATCH/swinv2-finetuned-openfake \
    --num_epochs 4 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_workers 4 \
    --cache_dir .cache
```

Notes:
- Real training image stats are expected at `../real_train_stats.npz`.
- This uses *Hugging Face* dataset streaming from *ComplexDataLab/OpenFake*.
- Results are logged to *Weights & Biases* if enabled.

## Evaluation

To evaluate a trained model checkpoint:

```bash
python evaluate.py \
    --resume_from_checkpoint /path/to/your/checkpoint \
    --cache_dir .cache
```

If using a `.safetensors` checkpoint, pass the full path:

```bash
--resume_from_checkpoint /path/to/checkpoint.safetensors
```

You can also apply LAION-style degradation to synthetic images before evaluation:

```bash
--degradation
```

The script reports:
- Per-model Accuracy, TPR, TNR
- Overall AUC-ROC, F1 Score, Accuracy

## Dataset

The scripts use the OpenFake dataset directly from *Hugging Face* in streaming mode, so no download is required beforehand.

### Additional Evaluation Sets

We provide two small evaluation sets for testing model generalization to out-of-distribution generators:

- [Imagen 3 test set (1,500 images)](https://drive.google.com/file/d/1hd-cfhkn2eTI6Aj-XdbNHa1M2vcbfjqa/view?usp=share_link)
- [Stable Diffusion 2.1 test set (1,500 images)](https://drive.google.com/file/d/1l4Om1ta28rZkqFxdaFm2DlEwKN19vzfM/view?usp=share_link)

These can be used to assess how well the model performs on unseen generators.

### Pretrained Model Checkpoint

A pretrained SwinV2 checkpoint trained on the OpenFake dataset is available here:

- [SwinV2 OpenFake checkpoint](https://drive.google.com/file/d/1tLWgcN-tbd2VcK_542ieIO5mwKGY8N9Y/view?usp=share_link)

## Example Output

Model: sd-3.5 — Acc: 0.9990, TPR: 0.9990, TNR: 0.0000  
Model: flux.1-dev — Acc: 0.9997, TPR: 0.9997, TNR: 0.0000  
Model: ideogram-3.0 — Acc: 0.9957, TPR: 0.9957, TNR: 0.0000  
Model: flux-1.1-pro — Acc: 0.9957, TPR: 0.9957, TNR: 0.0000  
Model: gpt-image-1 — Acc: 0.9957, TPR: 0.9957, TNR: 0.0000  
Model: real — Acc: 0.9955, TPR: 0.0000, TNR: 0.9955  
Overall — AUC-ROC: 0.9999, F1: 0.9963, Acc: 0.9963