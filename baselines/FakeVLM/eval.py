import os
import gc
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from sklearn.metrics import classification_report, accuracy_score


ImageFile.LOAD_TRUNCATED_IMAGES = True
TORCH_DTYPE = torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def load_model(device: str = "cuda") -> Tuple[LlavaForConditionalGeneration, AutoProcessor]:
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    # 1) Load base Florence-2
    model = LlavaForConditionalGeneration.from_pretrained(
        'lingcco/fakeVLM',
        trust_remote_code=True,
        torch_dtype=torch.float16,
        cache_dir='/home/mila/s/soroush.omranpour/scratch/hf_cache/'
    ).to(device_obj).eval()

    # 2) Processor from the fine-tuned FLODA repo
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", revision='a272c74')
    processor.patch_size = 14
    return model, processor


def predict_batch(model, processor, image_paths, device="cuda"):
    device_obj = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    images = []
    for p in image_paths:
        img = Image.open(p)
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        images.append(img.convert("RGB"))
    inputs = processor(text=['<image>Does the image look real/fake?'] * len(images),
                       images=images, return_tensors="pt", padding=True).to(device_obj)
    inputs = {k: (v.to(TORCH_DTYPE) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
              for k, v in inputs.items()}
    with torch.inference_mode():
        ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=10,
            num_beams=1,
            use_cache=True
        )
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    outs = []
    for t in texts:
        ans = t.split('?')[-1].split('.')[0].strip().lower()
        outs.append(ans)
    return outs

def normalize_prediction(raw_answer: str) -> str:
    if raw_answer is None:
        return "unknown"
    if 'real' in raw_answer:
        return "real"
    if 'fake' in raw_answer:
        return "fake"
    return "unknown"


def label_from_typ(typ: str) -> str:
    return "real" if str(typ).strip().lower() == "real" else "fake"


def run_inference_on_csv(
    in_csv: str,
    out_csv: str,
    root_dir: str,
    device: str = "cuda",
    save_every_n: int = 50,
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
    batch_size = 64  # adjust based on VRAM

    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i + batch_size]

        filenames = []
        typs = []
        for row in batch_rows:
            rel = str(row["filename"]).lstrip("/\\")
            filenames.append(os.path.join(root_dir, rel))
            typs.append(row.get("typ", "unknown"))

        try:
            raws = predict_batch(model, processor, filenames, device=device)
        except Exception as e:
            print(f"Error on batch starting at index {i}: {e}")
            raws = ["error"] * len(filenames)

        norms = [normalize_prediction(r) for r in raws]
        batch_df = pd.DataFrame({
            "filename": filenames,
            "typ": typs,
            "prediction": raws,
            "normalized_prediction": norms
        })
        results_df = pd.concat([results_df, batch_df], ignore_index=True)

        counter += len(filenames)
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
        df["label"] = df["label"].apply(lambda x: "real" if str(x).strip().lower() == "real" else "fake")
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
    print("\nClassification report (real vs fake):")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Accuracy: {acc:.4f}")
    return {"report": report, "accuracy": float(acc)}


def parse_args():
    parser = argparse.ArgumentParser(description="FLODA evaluation on OpenFake")
    parser.add_argument("--in_csv", type=str, required=True, help="Path to input CSV with 'filename' and optional 'typ'")
    parser.add_argument("--out_csv", type=str, default="floda_output_predictions.csv", help="Path to write predictions CSV")
    parser.add_argument("--root_dir", type=str, default=os.path.join("data", "OpenFake"), help="Root directory to prepend to relative filenames, e.g., data/OpenFake")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--save_every", type=int, default=64, help="Save progress every N images")
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
