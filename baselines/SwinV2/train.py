from datasets import load_dataset, DownloadConfig
import numpy as np
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from transformers import default_data_collator
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import argparse

import random
import cv2
from PIL import Image, ImageFile
import io
ImageFile.LOAD_TRUNCATED_IMAGES = True     # ← tolerate partial / corrupt files


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
    if random.random() < 0.2:
        target_h, target_w = random.choice(real_res_vals)
        orig_w, orig_h = img_pil.size
        orig_area = orig_w * orig_h
        target_area = target_h * target_w
        scale = (target_area / orig_area) ** 0.5
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))

        img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    img_np = np.array(img_pil)

    if random.random() < 0.2:
        target_blur = np.random.choice(real_blur_vals)
        blur_val = estimate_blur_laplacian(img_np)
        if blur_val > target_blur * 1.2:
            # GaussianBlur in OpenCV is highly optimized and releases the GIL
            img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.3, sigmaY=0.3)

    if random.random() < 0.2:
        # === Step 3: Add light Gaussian noise (OpenCV) ===
        sigma = int(255 * (noise_var ** 0.5))
        if sigma > 0:
            noise = np.zeros_like(img_np, dtype=np.int16)
            cv2.randn(noise, 0, sigma)             # in‑place Gaussian noise
            img_np = cv2.add(img_np.astype(np.int16), noise, dtype=cv2.CV_8U)

    if random.random() < 0.2:
        # === Step 4: Mild JPEG compression ===
        quality = np.random.randint(*jpeg_quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    return Image.fromarray(img_np)

cache_dir = os.path.join(os.environ["SCRATCH"], ".cache")
scratch_dir = os.environ.get("SCRATCH")
os.environ["WANDB_PROJECT"] = "SwinOpenFake"



def main(args):
    processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-small-patch4-window16-256", cache_dir=args.cache_dir, use_fast=True)
    model = AutoModelForImageClassification.from_pretrained("microsoft/swinv2-small-patch4-window16-256", cache_dir=args.cache_dir)
    
    # change the number of classes to 2
    model.num_labels = 2
    model.config.num_labels = 2
    model.classifier = torch.nn.Linear(model.swinv2.num_features, model.num_labels)
    model.to("cuda")
    
    real_train_stats = np.load('../real_train_stats.npz')
    real_blur_vals = real_train_stats['blur_vals']
    real_res_vals  = real_train_stats['res_vals']
    
    # Load streaming datasets from Hugging Face
    train_data = load_dataset("ComplexDataLab/OpenFake", split="train", streaming=True, download_config=DownloadConfig(cache_dir=args.cache_dir))
    eval_data = load_dataset("ComplexDataLab/OpenFake", split="test", streaming=True, download_config=DownloadConfig(cache_dir=args.cache_dir))

    # Preprocessing function to apply processor and degradation
    def preprocess_train(example):
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
        # Ensure 3‑channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_label = example["label"]
        # Map string labels to integer: 0=real, 1=fake
        if isinstance(raw_label, str):
            label = 0 if raw_label.lower() == "real" else 1
        else:
            label = int(raw_label)
        if label == 1:
            image = degrade_image_to_match_laion5(
                image, real_blur_vals, real_res_vals,
                seed=args.seed if hasattr(args, "seed") else None
            )
        inputs = processor(image, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "label": label}

    def preprocess_eval(example):
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
        # Ensure 3‑channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        raw_label = example["label"]
        # Map string labels to integer: 0=real, 1=fake
        if isinstance(raw_label, str):
            label = 0 if raw_label.lower() == "real" else 1
        else:
            label = int(raw_label)
        # No degradation on eval data
        inputs = processor(image, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "label": label}

    # Apply preprocessing to the streaming datasets
    train_data = train_data.map(preprocess_train)
    eval_data  = eval_data.map(preprocess_eval)

    # Metrics computation
    def compute_metrics(pred):
        logits = pred.predictions
        preds = logits.argmax(-1)
        labels = pred.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        auc_roc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc_roc": auc_roc,
        }
    
    max_steps = 600000 // args.batch_size * args.num_epochs

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=20,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        metric_for_best_model="f1",
        greater_is_better=True,
        max_steps=max_steps,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        run_name="swinv2-finetuned-openfake",
        report_to="wandb",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    return trainer, eval_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on OpenFake dataset")
    parser.add_argument("--output_dir", type=str, default="./swinv2-finetuned-openfake", help="Output directory for model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--num_workers", type=int, default=4,help="DataLoader worker processes")
    parser.add_argument("--cache_dir", type=str, default='.cache', help="Cache directory for datasets and models")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory or checkpoint name to resume training from"
    )
    args = parser.parse_args()
    trainer, eval = main(args)
    trainer.train(resume_from_checkpoint=False)
    #trainer._load_from_checkpoint(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # evaluate the model
    eval_results = trainer.evaluate(eval_dataset=eval)
    print(f"Evaluation results: {eval_results}")
