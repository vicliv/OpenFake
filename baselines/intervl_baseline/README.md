# InternVL Baseline for OpenFake Detection

This repository provides a baseline using the **InternVL** vision-language model for detecting AI-generated images in the [OpenFake](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) dataset.

---

## 📦 Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
````

Or install key dependencies directly:

```bash
pip install transformers timm pandas scikit-learn pillow
```

Make sure you also have PyTorch installed for your system. Visit [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally) for installation commands tailored to your hardware.

---

## 📂 Preparing OpenFake Dataset

### 1. Download OpenFake

Download the dataset from [Hugging Face](https://huggingface.co/datasets/CDL-AMLRT/OpenFake) or with:

```bash
wget https://huggingface.co/datasets/CDL-AMLRT/OpenFake/resolve/main/OpenFake.zip
unzip OpenFake.zip
```

### 2. Generate Label CSV

Use the `get_label_files.ipynb` notebook (provided in this repo) to generate a CSV file named `test_image_labels.csv`. This file must include:

* `filename`: Path to each image
* `typ`: Class name (`real`, `stylegan`, etc.)

Example:

```
filename,typ
OpenFake/test/real/img1234.jpg,real
OpenFake/test/sdxl/img5678.jpg,sdxl
```

---

## 🚀 Running Inference

Run the InternVL baseline with:

```bash
python vlm_baseline.py --in_csv test_image_labels.csv
```

This will create a file called `vlm_baseline_output_predictions.csv` with columns:

* `filename`
* `prediction` (e.g., "This image is real" or "This image is AI-generated")

---

## 📊 Evaluation

To evaluate model performance:

### 1. Normalize Predictions

The script `vlm_baseline.py` produces natural language outputs. Normalize them into categories:

```python
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("vlm_baseline_output_predictions.csv")

def normalize(pred):
    pred = pred.lower()
    if "real" in pred:
        return "real"
    elif "ai" in pred or "generated" in pred:
        return "ai-generated"
    return "unknown"

df["normalized_prediction"] = df["prediction"].apply(normalize)
df = df[df["normalized_prediction"] != "unknown"]

# Ground truth
df["label"] = df["typ"].apply(lambda x: "real" if x == "real" else "ai-generated")

y_true = df["label"]
y_pred = df["normalized_prediction"]

print(classification_report(y_true, y_pred, digits=4))
```

This will output precision, recall, F1-score, and accuracy for both real and AI-generated images.

---

## 📄 License

This project builds on InternVL from OpenGVLab. Please refer to the original license for usage terms.

---

## 🔗 References

* [OpenFake Dataset](https://huggingface.co/datasets/CDL-AMLRT/OpenFake)
* [InternVL on Hugging Face](https://huggingface.co/OpenGVLab)

```