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

## 📊 Baseline Results

| **Metric**           | **SwinV2-small** | **CLIP-D-10k+** | **Corvi2023** | **Fusion CLIP+Corvi** | **InternVL** |
|----------------------|------------------|------------------|----------------|------------------------|---------------|
| **Real Images TNR**  | **0.9949**       | 0.107            | 0.997          | 0.848                  | 0.589         |
|                      |                  |                  |                |                        |               |
| Ideogram 3.0         | 0.9987           | 0.947            | 0.176          | 0.400                  | 0.695         |
| GPT Image 1          | 0.9973           | 0.951            | 0.035          | 0.320                  | 0.766         |
| Flux 1.1-pro         | 0.9917           | 0.970            | 0.001          | 0.350                  | 0.609         |
| Flux 1.0-dev         | 0.9997           | 0.936            | 0.538          | 0.759                  | 0.627         |
| SDv3.5               | 0.9997           | 0.943            | 0.915          | 0.961                  | 0.637         |
| **Average TPR**      | **0.9974**       | 0.949            | 0.333          | 0.558                  | 0.667         |
|                      |                  |                  |                |                        |               |
| **Overall F1 Score** | **0.9956**       | 0.668            | 0.499          | 0.653                  | 0.642         |
| **Overall ROC AUC**  | **0.9956**       | 0.528            | 0.665          | 0.703                  | 0.628         |

> **Table**: Detection performance across five baselines. We report TNR on real images, TPR for each synthetic generator, and overall F1/ROC AUC.

---

## 🧠 Citation

```bibtex

````

---

## 🛡️ License

