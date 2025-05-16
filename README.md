# 🌀 OpenFake: An Open Dataset and Platform Toward Large-Scale Deepfake Detection

**OpenFake** is a continually updated benchmark for detecting AI-generated images. This repository contains:

- 📦 The [OpenFake dataset](https://huggingface.co/datasets/CDL-AMLRT/OpenFake)
- 🧠 Multiple detection baselines (CLIP-based, SwinV2, InternVL)
- 🖼️ Scripts to generate synthetic images using Stable Diffusion 3 and Flux

---

## 📁 Repository Structure

- `datasets/`: Tools and notebooks to build and label datasets (e.g., `get_label_files.ipynb`)
- `baselines/`: Includes SwinV2 (trainable), CLIP-based, and InternVL baselines  
  Each has its own README with instructions to reproduce results
---

## 📥 Downloads

We provide additional resources for evaluating model generalization to unseen generators:

- 📂 [Imagen 3 Test Set (~1,500 images)](https://drive.google.com/file/d/1hd-cfhkn2eTI6Aj-XdbNHa1M2vcbfjqa/view?usp=share_link)
- 📂 [Stable Diffusion 2.1 Test Set (~1,500 images)](https://drive.google.com/file/d/1l4Om1ta28rZkqFxdaFm2DlEwKN19vzfM/view?usp=share_link)

These can be used to benchmark detection performance on out-of-distribution samples.

---

## 📊 Baseline Results

| **Metric**           | **SwinV2-small** | **CLIP-D-10k+** | **Corvi2023** | **Fusion CLIP+Corvi** | **InternVL** |
|----------------------|------------------|------------------|----------------|------------------------|---------------|
| **Real Images TNR**  | **0.995**       | 0.107            | 0.997          | 0.848                  | 0.589         |
|                      |                  |                  |                |                        |               |
| Ideogram 3.0         | 0.999           | 0.947            | 0.176          | 0.400                  | 0.695         |
| GPT Image 1          | 0.997           | 0.951            | 0.035          | 0.320                  | 0.766         |
| Flux 1.1-pro         | 0.993           | 0.970            | 0.001          | 0.350                  | 0.609         |
| Flux 1.0-dev         | 1.000           | 0.936            | 0.538          | 0.759                  | 0.627         |
| SDv3.5               | 1.000           | 0.943            | 0.915          | 0.961                  | 0.637         |
| **Average TPR**      | **0.997**       | 0.949            | 0.333          | 0.558                  | 0.667         |
|                      |                  |                  |                |                        |               |
| **Overall F1 Score** | **0.997**       | 0.668            | 0.499          | 0.653                  | 0.642         |
| **Overall ROC AUC**  | **1.000**       | 0.528            | 0.665          | 0.703                  | 0.628         |

> **Table**: Detection performance across five baselines. We report TNR on real images, TPR for each synthetic generator, and overall F1/ROC AUC.

---

## 🧠 Citation

```bibtex

````

---

## 🛡️ License
