"""
Generates inpainted images using a diffusers inpainting pipeline.
Reads images and masks from a pre-staged temp directory, applies mask dilation,
saves output images to a staging directory, and appends metadata to a shared CSV.

Usage:
    Called automatically from inpaint_pipeline.py --> submit_generate_inpaint_samples.sh

    python generate_inpaint_samples.py MODEL_ID TOTAL_AMT_IMAGES_TO_GENERATE MODEL_TYPE BASE_MODEL_ID TEMP_DATA_DIR RELEASE_DATE
"""

import os
print(f"beginning imports...", flush=True)
import csv
import fcntl
import argparse
import torch
import pandas as pd
import numpy as np
import cv2
import gc
import traceback
from PIL import Image
from datetime import datetime
from diffusers import AutoPipelineForInpainting
from dotenv import load_dotenv
print(f"imports done.", flush=True)

load_dotenv()

# ---- Arguments ----
parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("total_images", type=int)
parser.add_argument("model_type")
parser.add_argument("base_model_id")
parser.add_argument("temp_data_dir")
parser.add_argument("release_date")
parser.add_argument("--staging-dir", default="data/staging_inpaint_images")
parser.add_argument("--masks-dir", default="data/open_images/masks")
args = parser.parse_args()

MODEL_ID = args.model_id
TOTAL_AMT_IMAGES_TO_GENERATE = args.total_images
MODEL_TYPE = args.model_type
BASE_MODEL_ID = args.base_model_id
TEMP_DATA_DIR = args.temp_data_dir
RELEASE_DATE = args.release_date

STAGING_DIR = args.staging_dir
METADATA_CSV = os.path.join(STAGING_DIR, "metadata.csv")

IMAGE_BASE_URL = "https://s3.amazonaws.com/open-images-dataset/train"
MASK_BASE_URL = "https://storage.googleapis.com/openimages/v5/train-masks/train-masks-"

METADATA_FIELDS = [
    "filename", "prompt", "label", "model", "type", "release_date",
    "source_image_url", "mask_url", "mask_path", "class_name",
]

# Error codes (same as txt2img pipeline)
EXIT_DATA_FAULT = 10
EXIT_MODEL_FAULT = 11
EXIT_MEMORY_FAULT = 12
EXIT_CODE_BUG = 14

# SLURM array task info
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32


import random

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def dilate_mask(mask_pil, image_size):
    """
    Dilate a mask with randomized parameters for variety:
    - Dilation amount: random between 2-8% of image's shorter dimension
    - Kernel shape: randomly ellipse, rectangle, or cross
    - Optionally apply slight gaussian blur for soft edges
    """
    mask_np = np.array(mask_pil)
    shorter_dim = min(image_size)

    # Random dilation: 2-8% of shorter image dimension
    dilation_px = random.randint(int(shorter_dim * 0.02), int(shorter_dim * 0.08))
    dilation_px = max(3, dilation_px)  # minimum 3px

    # Random kernel shape
    kernel_type = random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(kernel_type, (dilation_px * 2 + 1, dilation_px * 2 + 1))

    dilated = cv2.dilate(mask_np, kernel, iterations=1)

    # 50% chance of slight blur for softer edges
    if random.random() < 0.5:
        blur_size = random.choice([3, 5, 7])
        dilated = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
        # Re-threshold to keep it binary-ish (inpainting pipelines expect this)
        _, dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)

    return Image.fromarray(dilated)


MASKS_DIR = args.masks_dir


def load_image_and_mask(row):
    """
    Load image from temp dir (downloaded per model run) and mask from
    permanent storage (pre-extracted from Open Images zips).
    """
    image_id = row["ImageID"]
    mask_filename = row["MaskPath"]

    img_path = os.path.join(TEMP_DATA_DIR, "images", f"{image_id}.jpg")
    mask_path = os.path.join(MASKS_DIR, mask_filename)

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # Resize mask to match image if needed
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)

    mask = dilate_mask(mask, image.size)

    return image, mask


def append_metadata_row(row_dict):
    """Append a single metadata row to the shared CSV with file locking."""
    file_exists = os.path.exists(METADATA_CSV)
    with open(METADATA_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists or os.path.getsize(METADATA_CSV) == 0:
            writer.writeheader()
        writer.writerow(row_dict)
        fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
try:
    # Load the manifest — the pipeline already sampled and verified masks
    MANIFEST_CSV = os.path.join(TEMP_DATA_DIR, "batch_manifest.csv")
    model_df = pd.read_csv(MANIFEST_CSV)

    images_per_task = len(model_df) // task_count
    start_index = task_id * images_per_task
    end_index = len(model_df) if task_id == task_count - 1 else start_index + images_per_task
    target_count_for_this_gpu = end_index - start_index

    task_df = model_df.iloc[start_index:end_index].reset_index(drop=True)

    print(
        f"Task {task_id}: Processing {target_count_for_this_gpu} images "
        f"(rows {start_index}-{end_index} of {TOTAL_AMT_IMAGES_TO_GENERATE}). "
        f"Shuffled from {len(model_df)} total prompts.",
        flush=True,
    )

    # Ensure output directory exists
    os.makedirs(STAGING_DIR, exist_ok=True)

    # ---- Load pipeline ----
    if MODEL_TYPE == "LoRA":
        print(
            f"Task {task_id}: Loading base model ({BASE_MODEL_ID}) for LoRA injection...",
            flush=True,
        )
        pipeline = AutoPipelineForInpainting.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        print(f"Task {task_id}: Injecting LoRA weights from {MODEL_ID}...", flush=True)
        pipeline.load_lora_weights(MODEL_ID, local_files_only=True)
    else:
        print(
            f"Task {task_id}: Loading standalone model ({MODEL_ID})...", flush=True
        )
        pipeline = AutoPipelineForInpainting.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )

    # ---- Offloading & memory optimization ----
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload()
        print(f"Task {task_id}: CPU offloading enabled.", flush=True)
    else:
        pipeline = pipeline.to(device)
        print(f"Task {task_id}: Model manually moved to GPU.", flush=True)

    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print(f"Task {task_id}: VAE slicing enabled.", flush=True)

    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        print(f"Task {task_id}: VAE tiling enabled.", flush=True)

    # ---- Generation loop ----
    optimal_batch_size = 4  # Inpainting uses more VRAM than txt2img, start smaller
    images_generated = 0
    today_date = datetime.now().strftime("%Y-%m-%d")
    safe_model_name = MODEL_ID.replace("/", "_")

    while images_generated < target_count_for_this_gpu:
        current_chunk_size = min(
            optimal_batch_size, target_count_for_this_gpu - images_generated
        )
        batch_slice = task_df.iloc[images_generated : images_generated + current_chunk_size]

        try:
            # Load and prepare batch
            batch_images = []
            batch_masks = []
            batch_prompts = []

            for _, row in batch_slice.iterrows():
                img, mask = load_image_and_mask(row)
                batch_images.append(img)
                batch_masks.append(mask)
                batch_prompts.append(row.get("generated_prompt", "a realistic replacement"))

            results = pipeline(
                prompt=batch_prompts,
                image=batch_images,
                mask_image=batch_masks,
            )

            for j, img in enumerate(results.images):
                global_index = start_index + images_generated + j
                filename = f"inpaint_{safe_model_name}_{today_date}_{global_index}.png"

                # Save image to staging directory
                img.save(os.path.join(STAGING_DIR, filename))

                row_data = batch_slice.iloc[j]
                image_id = row_data["ImageID"]
                mask_filename = row_data["MaskPath"]
                mask_shard = mask_filename[0].lower()

                # Append metadata row to shared CSV
                append_metadata_row({
                    "filename": filename,
                    "prompt": batch_prompts[j],
                    "label": "fake",
                    "model": MODEL_ID,
                    "type": MODEL_TYPE,
                    "release_date": RELEASE_DATE,
                    "source_image_url": f"{IMAGE_BASE_URL}/{image_id}.jpg",
                    "mask_url": f"{MASK_BASE_URL}{mask_shard}/{mask_filename}",
                    "mask_path": mask_filename,
                    "class_name": row_data["ClassName"],
                })

            images_generated += current_chunk_size

            if images_generated % 100 < optimal_batch_size and images_generated > 0:
                print(
                    f"Task {task_id} generated {images_generated}/{target_count_for_this_gpu}...",
                    flush=True,
                )

        except torch.cuda.OutOfMemoryError:
            print(
                f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing...",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()

            if optimal_batch_size == 1:
                print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                sys.exit(EXIT_MEMORY_FAULT)

            optimal_batch_size = max(1, optimal_batch_size // 2)

        except Exception as inner_e:
            if "cuda out of memory" in str(inner_e).lower() or "oom" in str(inner_e).lower():
                print(
                    f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing...",
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()

                if optimal_batch_size == 1:
                    print(
                        f"Task {task_id} FATAL: Model too large for 1 image.", flush=True
                    )
                    sys.exit(EXIT_MEMORY_FAULT)

                optimal_batch_size = max(1, optimal_batch_size // 2)
            else:
                raise inner_e

    print(f"Success! Task {task_id} finished inpainting for {MODEL_ID}", flush=True)

except Exception as e:
    error_msg = str(e)
    error_msg_lower = error_msg.lower()
    print(f"Task {task_id} ERROR: {traceback.format_exc()}", flush=True)

    if "cuda out of memory" in error_msg_lower:
        sys.exit(EXIT_MEMORY_FAULT)

    elif isinstance(e, (FileNotFoundError, OSError)) or "no such file or directory" in error_msg_lower:
        hf_keywords = [
            "model_index.json", "config.json", "scheduler",
            "huggingface", "safetensors",
        ]
        if any(kw in error_msg_lower for kw in hf_keywords):
            sys.exit(EXIT_MODEL_FAULT)
        else:
            sys.exit(EXIT_DATA_FAULT)

    elif any(
        kw in error_msg_lower
        for kw in [
            "weight", "pipeline", "expected str",
            "time embedding", "incorrect config", "dimension",
        ]
    ):
        sys.exit(EXIT_MODEL_FAULT)

    else:
        print(
            f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True
        )
        sys.exit(EXIT_CODE_BUG)
