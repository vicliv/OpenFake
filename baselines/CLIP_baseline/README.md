# CLIP-Based Synthetic Image Detection (Modified for OpenFake)

This repository is a modified version of the official [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) baseline, adapted to run on the [OpenFake](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) dataset.

The original method is described in the paper:

**Raising the Bar of AI-generated Image Detection with CLIP**  
*Davide Cozzolino, Giovanni Poggi, Riccardo Corvi, Matthias Nießner, and Luisa Verdoliva*  
[arXiv:2312.00195v2](https://arxiv.org/abs/2312.00195v2)

---

## 💡 Summary

The original repository does not include training code, but provides pre-trained weights and inference scripts. This version adds compatibility with the OpenFake dataset to use the model as a plug-and-play baseline.

---

## 📦 Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
````

Required packages include:

* `torch`, `torchvision`, `timm>=0.9.10`
* `open_clip_torch`, `huggingface-hub>=0.23.0`
* `pandas`, `scikit-learn`, `tqdm`, `pillow`, `yaml`

If not already done, pull the pretrained weights:

```bash
git lfs pull
```

---

## 📂 Preparing OpenFake Dataset

To run this model on the [OpenFake](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) dataset:

### 1. Download OpenFake

Download from [Hugging Face](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) or use:

```bash
wget https://huggingface.co/datasets/CDL-AMLRT/OpenFake/resolve/main/OpenFake.zip
unzip OpenFake.zip
```

### 2. Generate CSV File

Use the `get_label_files.ipynb` notebook (provided in this repo) to generate a CSV file named `test_image_labels.csv`. This file must contain:

* `filename`: Path to the image
* `typ`: Class name (e.g., `real`, `stylegan`, etc.)

Example:

```
filename,typ
OpenFake/test/real/img1234.jpg,real
OpenFake/test/sdxl/img5678.jpg,sdxl
```

---

## 🚀 Inference

Run inference on the labeled image list:

```bash
python main.py --in_csv test_image_labels.csv
```

This will output `results.csv`

---

## 📊 Evaluation

To compute metrics like AUC and accuracy:

```bash
python compute_metrics.py --in_csv test_image_labels.csv
```

This uses the ground truth in `test_image_labels.csv` and predictions from `results.csv`.

---

## 📄 License

This project is based on work by GRIP-UNINA and is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).

---

## 🔗 Citation

If you use this code or model, please cite the original paper:

```bibtex
@inproceedings{cozzolino2023raising,
  author={Davide Cozzolino and Giovanni Poggi and Riccardo Corvi and Matthias Nießner and Luisa Verdoliva},
  title={{Raising the Bar of AI-generated Image Detection with CLIP}}, 
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
  year={2024},
}
```
