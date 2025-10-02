import os
import gc
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from sklearn.metrics import classification_report, accuracy_score

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Workaround for optional flash_attn import in Florence-2 dynamic module
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


# Default dtype and device for FLODA evaluation (float16 for efficiency)
TORCH_DTYPE = torch.float16


def load_model(device: str = "cuda") -> Tuple[AutoModelForCausalLM, AutoProcessor]:
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    # 1) Load base Florence-2
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            'microsoft/Florence-2-base-ft',
            trust_remote_code=True,
            torch_dtype=TORCH_DTYPE,
        ).to(device_obj)

    # 2) Processor from the fine-tuned FLODA repo
    processor = AutoProcessor.from_pretrained('byh711/FLODA-deepfake', trust_remote_code=True)

    # 3) Apply LoRA adapters from FLODA
    model = PeftModel.from_pretrained(model, 'byh711/FLODA-deepfake')

    # 4) Merge adapters for faster inference
    model = model.merge_and_unload()
    model.eval()
    return model, processor


def predict_single(model, processor, image_path: str, device: str = "cuda") -> str:
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    image = Image.open(image_path)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    task_prompt = "<DEEPFAKE_DETECTION>"
    text_input = "Is this photo real?"
    prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device_obj)
    # Ensure floating tensors are in the expected dtype
    inputs = {k: (v.to(TORCH_DTYPE) if isinstance(v, torch.Tensor) and v.is_floating_point() else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )[task_prompt]

    return str(result).strip()


def normalize_prediction(raw_answer: str) -> str:
    if raw_answer is None:
        return "unknown"
    text = str(raw_answer).strip().lower()
    # FLODA is expected to answer yes/no to "Is this photo real?"
    if any(tok in text for tok in ["yes", "real", "true"]):
        return "real"
    if any(tok in text for tok in ["no", "fake", "ai", "generated", "ai-generated", "synth"]):
        return "ai-generated"
    return "unknown"


def label_from_typ(typ: str) -> str:
    return "real" if str(typ).strip().lower() == "real" else "ai-generated"


def run_inference_on_csv(
    in_csv: str,
    out_csv: str,
    root_dir: str,
    device: str = "cuda",
    save_every_n: int = 20,
    compute_metrics: bool = True
) -> Optional[dict]:
    df = pd.read_csv(in_csv)
    # Accept either 'filename' or 'file_name' and normalize to 'filename'
    if "filename" not in df.columns and "file_name" in df.columns:
        df["filename"] = df["file_name"]
    assert "filename" in df.columns, "CSV must have a 'filename' or 'file_name' column"

    # Optional: carry through typ if present; otherwise derive from 'label' if available
    if "typ" not in df.columns:
        if "label" in df.columns:
            df["typ"] = df["label"].apply(lambda x: "real" if str(x).strip().lower() == "real" else "fake")
        else:
            df["typ"] = "unknown"

    already_done = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        if not existing.empty and "filename" in existing.columns:
            already_done = set(existing["filename"].tolist())
        results_df = existing
    else:
        results_df = pd.DataFrame(columns=["filename", "typ", "prediction", "normalized_prediction"])

    df = df[~df["filename"].isin(already_done)]
    print(f"Processing {len(df)} images...")
    if df.empty:
        print("No new images to process.")
        if compute_metrics and os.path.exists(out_csv):
            return evaluate_predictions(out_csv)
        return None

    model, processor = load_model(device=device)

    counter = 0
    for _, row in df.iterrows():
        rel = str(row["filename"]).lstrip("/\\")
        filename = os.path.join(root_dir, rel)
        typ = row.get("typ", "unknown")
        try:
            with torch.no_grad():
                raw = predict_single(model, processor, filename, device=device)
        except Exception as e:
            print(f"Error on {filename}: {e}")
            raw = "error"

        norm = normalize_prediction(raw)
        results_df = pd.concat([
            results_df,
            pd.DataFrame([(filename, typ, raw, norm)], columns=["filename", "typ", "prediction", "normalized_prediction"])
        ], ignore_index=True)

        counter += 1
        if counter % save_every_n == 0:
            results_df.to_csv(out_csv, index=False)
            print(f"Saved progress at {counter} images → {out_csv}")
            torch.cuda.empty_cache()
            gc.collect()

    results_df.to_csv(out_csv, index=False)
    print(f"Saved all predictions to {out_csv}")

    if compute_metrics:
        return evaluate_predictions(out_csv)
    return None


def evaluate_predictions(pred_csv: str) -> dict:
    df = pd.read_csv(pred_csv)
    if df.empty:
        print("No predictions to evaluate.")
        return {}

    if "normalized_prediction" not in df.columns:
        df["normalized_prediction"] = df["prediction"].apply(normalize_prediction)

    # Prefer ground-truth from 'label' if present; otherwise derive from 'typ'
    if "label" in df.columns:
        df["label"] = df["label"].apply(lambda x: "real" if str(x).strip().lower() == "real" else "ai-generated")
    else:
        df["label"] = df["typ"].apply(label_from_typ)
    df = df[df["normalized_prediction"] != "unknown"]
    if df.empty:
        print("All predictions are unknown; cannot compute metrics.")
        return {}

    y_true = df["label"].tolist()
    y_pred = df["normalized_prediction"].tolist()

    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    acc = accuracy_score(y_true, y_pred)
    print("\nClassification report (real vs ai-generated):")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Accuracy: {acc:.4f}")
    return {"report": report, "accuracy": float(acc)}


def parse_args():
    parser = argparse.ArgumentParser(description="FLODA evaluation on OpenFake")
    parser.add_argument("--in_csv", type=str, required=True, help="Path to input CSV with 'filename' and optional 'typ'")
    parser.add_argument("--out_csv", type=str, default="floda_output_predictions.csv", help="Path to write predictions CSV")
    parser.add_argument("--root_dir", type=str, default=os.path.join("data", "OpenFake"), help="Root directory to prepend to relative filenames, e.g., data/OpenFake")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--save_every", type=int, default=20, help="Save progress every N images")
    parser.add_argument("--no_metrics", action="store_true", help="Disable metrics computation after inference")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference_on_csv(
        in_csv=os.path.expanduser(args.in_csv),
        out_csv=os.path.expanduser(args.out_csv),
        root_dir=os.path.expanduser(args.root_dir),
        device=args.device,
        save_every_n=args.save_every,
        compute_metrics=not args.no_metrics,
    )
