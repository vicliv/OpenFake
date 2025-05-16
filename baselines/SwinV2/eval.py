from datasets import load_dataset
import os
import argparse
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from safetensors.torch import load_file as safe_load_file
import pandas as pd

import io
from collections import defaultdict
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def estimate_blur_laplacian(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def degrade_image_to_match_laion5(img_pil, real_blur_vals, real_res_vals,
                                  noise_var=0.0005, jpeg_quality_range=(70, 95), seed=None):
    """
    Lightly degrades an image to mimic real training images' resolution and blur distribution.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # === Step 1: Resize to match real training image resolution ===
    target_h, target_w = random.choice(real_res_vals)
    orig_w, orig_h = img_pil.size
    orig_area = orig_w * orig_h
    target_area = target_h * target_w
    scale = (target_area / orig_area) ** 0.5
    new_w = max(1, int(orig_w * scale))
    new_h = max(1, int(orig_h * scale))

    img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img_pil)

    target_blur = np.random.choice(real_blur_vals)
    blur_val = estimate_blur_laplacian(img_np)
    if blur_val > target_blur * 1.2:
        # GaussianBlur in OpenCV is highly optimized and releases the GIL
        img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.3, sigmaY=0.3)

    # === Step 3: Add light Gaussian noise (OpenCV) ===
    sigma = int(255 * (noise_var ** 0.5))
    if sigma > 0:
        noise = np.zeros_like(img_np, dtype=np.int16)
        cv2.randn(noise, 0, sigma)             # in‑place Gaussian noise
        img_np = cv2.add(img_np.astype(np.int16), noise, dtype=cv2.CV_8U)

    # === Step 4: Mild JPEG compression ===
    quality = np.random.randint(*jpeg_quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img_np, encode_param)
    img_np = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    return Image.fromarray(img_np)

# === Resplit logic for real images based on baseline model false positives ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resplit real images based on baseline model false positives")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to checkpoint to load baseline model from")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument("--cache_dir", type=str, default='.cache', help="Cache directory for datasets and models")
    parser.add_argument('--degradation', action='store_true', help="Apply degradation to images")
    args = parser.parse_args()

    # Load processor and baseline model
    processor = AutoImageProcessor.from_pretrained(
        "microsoft/swinv2-small-patch4-window16-256", cache_dir=args.cache_dir, use_fast=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        "microsoft/swinv2-small-patch4-window16-256", cache_dir=args.cache_dir
    )
    model.num_labels = 2
    model.config.num_labels = 2
    model.classifier = torch.nn.Linear(model.swinv2.num_features, model.num_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Load checkpoint if provided
    if args.resume_from_checkpoint:
        ckpt_path = args.resume_from_checkpoint
        # Load safetensor checkpoint if provided
        if ckpt_path.endswith(".safetensors"):
            # Load safetensors checkpoint on CPU, then move to target device
            state = safe_load_file(ckpt_path, device='cpu')
        else:
            # Expect a directory containing pytorch_model.bin
            state = torch.load(os.path.join(ckpt_path, "pytorch_model.bin"), map_location=device)
        model.load_state_dict(state, False)
    model.to(device)
    
    real_train_stats = np.load('../real_train_stats.npz')
    real_blur_vals = real_train_stats['blur_vals']
    real_res_vals  = real_train_stats['res_vals']
    
    # === Load the OpenFake test split in streaming mode ===
    dataset = load_dataset(
        "ComplexDataLab/OpenFake",
        split="test",
        streaming=True,
    )

    # Pre‑processing helper (similar to train.py preprocess_eval)
    def preprocess_example(example):
        image = example["image"]
        # Ensure image is a PIL Image in RGB mode
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, (bytes, bytearray)):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, dict):
                if "bytes" in image and image["bytes"] is not None:
                    image = Image.open(io.BytesIO(image["bytes"]))
                elif "path" in image and image["path"]:
                    image = Image.open(image["path"])
                else:
                    raise ValueError(f"Unsupported image dict keys: {image.keys()}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")

        raw_label = example["label"]
        # Map string labels to integer
        if isinstance(raw_label, str):
            label = 0 if raw_label.lower() == "real" else 1
        else:
            label = int(raw_label)
        
        if label == 1 and args.degradation:
            image = degrade_image_to_match_laion5(
                image, real_blur_vals, real_res_vals,
                seed=args.seed if hasattr(args, "seed") else None
            )
        model_name = example.get("model")

        inputs = processor(image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return {"pixel_values": pixel_values, "label": label, "model_name": model_name}

    dataset = dataset.map(preprocess_example)

    batch_size = 32
    buffer_pixels, buffer_labels, buffer_models = [], [], []

    metrics = defaultdict(lambda: {"TP":0,"TN":0,"FP":0,"FN":0})
    all_labels, all_preds, all_scores = [], [], []

    def process_batch():
        if len(buffer_pixels) == 0:
            return
        pv = torch.stack(buffer_pixels).to(device)
        with torch.no_grad():
            outputs = model(pv)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=1)[:,1].cpu().numpy()
        # record metrics
        for pred, lbl, mdl, prob in zip(preds, buffer_labels, buffer_models, probs):
            m = metrics[mdl]
            if lbl == 1:
                if pred == 1:
                    m["TP"] += 1
                else:
                    m["FN"] += 1
            else:
                if pred == 0:
                    m["TN"] += 1
                else:
                    m["FP"] += 1
            all_labels.append(lbl)
            all_preds.append(int(pred))
            all_scores.append(float(prob))
        buffer_pixels.clear(); buffer_labels.clear(); buffer_models.clear()

    # Iterate over the streaming dataset
    from itertools import islice
    for example in tqdm(dataset):
        buffer_pixels.append(example["pixel_values"])
        buffer_labels.append(example["label"])
        buffer_models.append(example["model_name"])
        if len(buffer_pixels) >= batch_size:
            process_batch()
    # process any remaining
    process_batch()

    # Compute and print per-model metrics
    for model_name, m in metrics.items():
        TP, TN, FP, FN = m["TP"], m["TN"], m["FP"], m["FN"]
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total > 0 else 0.0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        print(f"Model: {model_name} — Acc: {accuracy:.4f}, TPR: {TPR:.4f}, TNR: {TNR:.4f}")

    # Compute and print overall metrics
    overall_auc = roc_auc_score(all_labels, all_scores)
    overall_f1 = f1_score(all_labels, all_preds)
    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"Overall — AUC-ROC: {overall_auc:.4f}, F1: {overall_f1:.4f}, Acc: {overall_acc:.4f}")