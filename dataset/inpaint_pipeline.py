"""
Inpainting HuggingFace pipeline code. Start it with `./start_inpaint_pipeline.sh`

Usage:
    python -u inpaint_pipeline.py                              # scan all diffusers models
    python -u inpaint_pipeline.py --model black-forest-labs/FLUX.1-dev  # run specific model(s)
    python -u inpaint_pipeline.py --model model/A model/B      # run multiple specific models

For each compatible inpainting model:
1. Downloads model weights (login node, online)
2. Downloads only the required IMAGES from S3 (masks are already on disk)
3. Submits a SLURM array job to the offline compute nodes for generation
4. Cleans up temp images to free file quota (masks stay permanently)
"""

import os
import json
import pandas as pd
import subprocess
import shutil
import time
import sys
import argparse
import concurrent.futures
from dotenv import load_dotenv
import builtins
from datetime import datetime

load_dotenv()

from huggingface_hub import HfApi, model_info as get_model_info, snapshot_download

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REGISTRY_FILE = "inpaint_model_registry.json"
STAGING_DIR = "data/staging_inpaint_images"
MASTER_PROMPTS_CSV = "data/open_images/master_prompts_300k.csv"
MASKS_DIR = "data/open_images/masks"
INPAINT_TMP_ROOT = "data/inpaint_tmp"
INPAINT_SCRIPT = "submit_generate_inpaint_samples.sh"
INPAINT_LARGE_SCRIPT = "submit_generate_inpaint_samples_large.sh"
SLURM_LOG_DIR = "data/slurm_logs"
TODAY = datetime.now().strftime("%Y-%m-%d")

IMAGE_BASE_URL = "https://s3.amazonaws.com/open-images-dataset/train"

# Parallel download threads for images (I/O-bound)
DOWNLOAD_WORKERS = 32

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    builtins.print(timestamp, *args, **kwargs)

print = timestamped_print

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------
def load_registry():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

# ---------------------------------------------------------------------------
# Model inspection helpers
# ---------------------------------------------------------------------------
def check_safetensors_available(model_id):
    print("Verifying safetensors...")
    try:
        info = get_model_info(model_id)
        print(f"Scanning through {len(info.siblings)} items")
        file_names = [f.rfilename for f in info.siblings]
        return any(fname.endswith(".safetensors") for fname in file_names)
    except Exception as e:
        print(f"Error checking files for {model_id}: {e}")
        return False


def classify_model(model):
    """
    Returns (model_type, target_count, base_model_id).
    target_count determines how many inpainting samples to generate:
        Base:      10k
        Fine-tune: 5k
        LoRA:      2k
    """
    tags = [tag.lower() for tag in (model.tags or [])]
    card_data = getattr(model, "cardData", {}) or {}
    base_model = card_data.get("base_model", None)

    if isinstance(base_model, list) and len(base_model) > 0:
        base_model = base_model[1]

    if not base_model:
        for tag in tags:
            if tag.startswith("base_model:"):
                base_model = tag.split(":", 1)
                break

    safe_base = base_model if base_model else "None"

    if "lora" in tags or "peft" in tags or "adapter" in tags:
        return "LoRA", 2000, safe_base[1] if isinstance(safe_base, list) else safe_base

    if base_model:
        return "Fine-tune", 5000, safe_base[1] if isinstance(safe_base, list) else safe_base

    return "Base", 10000, "None"

# ---------------------------------------------------------------------------
# Image download (masks are permanently on disk at MASKS_DIR)
# ---------------------------------------------------------------------------
def load_master_prompts():
    df = pd.read_csv(MASTER_PROMPTS_CSV)
    return df


def download_single_image(args):
    image_id, dest_path = args
    if os.path.exists(dest_path):
        return True
    url = f"{IMAGE_BASE_URL}/{image_id}.jpg"
    result = subprocess.run(
        ["curl", "-sS", "-L", "-f", "-o", dest_path, "--max-time", "30", url],
        capture_output=True, text=True
    )
    return result.returncode == 0


def download_images(batch_df, temp_images_dir):
    """Download images from S3 in parallel. Masks are NOT downloaded."""
    os.makedirs(temp_images_dir, exist_ok=True)

    # Deduplicate by ImageID
    unique_images = {}
    for _, row in batch_df.iterrows():
        image_id = row["ImageID"]
        if image_id not in unique_images:
            unique_images[image_id] = os.path.join(temp_images_dir, f"{image_id}.jpg")

    to_download = [(iid, path) for iid, path in unique_images.items() if not os.path.exists(path)]
    already = len(unique_images) - len(to_download)
    print(f"Downloading {len(to_download)} images ({already} already cached)...")

    if not to_download:
        return 0

    failed = 0
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_single_image, item): item for item in to_download}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            if not future.result():
                failed += 1
            if done % 500 == 0:
                print(f"  Downloaded {done}/{len(to_download)} images...", flush=True)

    print(f"Download complete. {len(to_download) - failed} succeeded, {failed} failed.")
    return failed


def get_valid_sample(master_df, target_count, masks_dir, model_id):
    """
    Shuffles the master dataframe with a model-specific seed and verifies masks lazily.
    Each model gets a different random selection from the full 300k.
    Stops as soon as it finds enough valid masks.
    """
    shuffled_df = master_df.sample(
        frac=1, random_state=hash(model_id) % 2**31
    ).reset_index(drop=True)

    valid_rows = []
    missing = 0

    for _, row in shuffled_df.iterrows():
        mask_path = os.path.join(masks_dir, row["MaskPath"])

        if os.path.exists(mask_path):
            valid_rows.append(row)
            if len(valid_rows) == target_count:
                break
        else:
            missing += 1

    if missing > 0:
        print(f"WARNING: Encountered {missing} missing masks during sampling.")

    return pd.DataFrame(valid_rows)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_inpaint_pipeline(specific_models=None):
    api = HfApi()
    registry = load_registry()

    # Load the master prompts CSV once
    master_df = load_master_prompts()
    print(f"Loaded {len(master_df)} rows from master prompts CSV.")

    if specific_models:
        # Run specific models — fetch their info from HF
        print(f"Running specific models: {specific_models}")
        models = []
        for mid in specific_models:
            try:
                info = get_model_info(mid)
                models.append(info)
                print(f"  Found: {mid}")
            except Exception as e:
                print(f"  Could not find {mid}: {e}")
    else:
        print("Fetching inpainting-capable models from Hugging Face...")
        models = api.list_models(filter="diffusers", sort="downloads", full=True, limit=1000)

    for model in models:
        # Skip filters when running specific models
        if not specific_models:
            if model.id in registry:
                status = registry[model.id].get("status")
                if status in ["COMPLETED", "MODEL_FAULT"]:
                    continue

            downloads = getattr(model, "downloads", 0)
            if downloads < 30000:
                continue

            pipeline_task = getattr(model, "pipeline_tag", "") or ""
            if any(bad in pipeline_task.lower() for bad in ["audio", "3d", "video"]):
                print(f"Skipping {model.id}: incompatible task '{pipeline_task}'.")
                registry[model.id] = {
                    "status": "MODEL_FAULT",
                    "reason": f"Incompatible task: {pipeline_task}",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

            if not check_safetensors_available(model.id):
                registry[model.id] = {
                    "status": "MODEL_FAULT",
                    "reason": "No safetensors found",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

        model_type, target_count, base_model_id = classify_model(model)
        safe_model_name = model.id.replace("/", "_")
        release_date = model.created_at.strftime("%Y-%m-%d") if getattr(model, "created_at", None) else "unknown"

        print(f"\n--- Processing {model.id} ({model_type}, target={target_count}) ---")

        try:
            # ---------------------------------------------------------------
            # Step 1: Download model weights
            # ---------------------------------------------------------------
            print(f"Downloading model weights for {model.id}...")
            snapshot_download(repo_id=model.id)

            if model_type == "LoRA" and base_model_id != "None":
                print(f"Downloading base model weights for {base_model_id}...")
                snapshot_download(repo_id=base_model_id)
            elif model_type == "LoRA" and base_model_id == "None":
                registry[model.id] = {
                    "status": "MODEL_FAULT",
                    "reason": "Missing base model dependency",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

            # ---------------------------------------------------------------
            # Step 2: Sample prompts, verify masks, download images
            # ---------------------------------------------------------------
            temp_data_dir = os.path.join(INPAINT_TMP_ROOT, safe_model_name)
            temp_images_dir = os.path.join(temp_data_dir, "images")

            # Get exactly target_count valid rows with model-specific shuffle
            sampled_df = get_valid_sample(master_df, target_count, MASKS_DIR, model.id)

            if len(sampled_df) < target_count:
                print(f"ERROR: Could only find {len(sampled_df)} valid masks, need {target_count}. "
                      f"Skipping {model.id}. Run download_masks.py to fix.")
                registry[model.id] = {
                    "status": "INFRASTRUCTURE_FAULT",
                    "reason": f"Only {len(sampled_df)} masks available, need {target_count}",
                    "date": TODAY,
                }
                save_registry(registry)
                continue

            # Save manifest so compute script knows exactly which rows to use
            os.makedirs(temp_data_dir, exist_ok=True)
            sampled_df.to_csv(
                os.path.join(temp_data_dir, "batch_manifest.csv"), index=False
            )

            print(f"Downloading {len(sampled_df)} images to {temp_images_dir}...")
            download_images(sampled_df, temp_images_dir)

            # ---------------------------------------------------------------
            # Step 3: Submit SLURM job with tiered resources
            # ---------------------------------------------------------------
            if target_count >= 10000:
                array_tasks = 10
            elif target_count >= 5000:
                array_tasks = 5
            else:
                array_tasks = 3

            # Check if this is a retry (previous INFRASTRUCTURE_FAULT)
            prev_status = registry.get(model.id, {})
            is_retry = prev_status.get("status") == "INFRASTRUCTURE_FAULT"
            was_oom = "Memory" in prev_status.get("reason", "")

            if is_retry and was_oom:
                sbatch_script = INPAINT_LARGE_SCRIPT
                print(f"RETRY (OOM) — using large resource allocation.")
            else:
                sbatch_script = INPAINT_SCRIPT

            log_dir = f"{SLURM_LOG_DIR}/{TODAY}"
            os.makedirs(log_dir, exist_ok=True)

            print(f"Submitting inpaint array ({array_tasks} tasks) via {sbatch_script}...")

            process = subprocess.run(
                [
                    "sbatch",
                    "--wait",
                    f"--array=0-{array_tasks - 1}",
                    f"--output={log_dir}/gen_inpaint-{safe_model_name}-%A_%a.out",
                    sbatch_script,
                    model.id,
                    str(target_count),
                    model_type,
                    base_model_id,
                    temp_data_dir,
                    release_date,
                ],
                capture_output=True,
                text=True,
            )

            # Wait for Lustre to flush
            time.sleep(10)

            if process.returncode == 0:
                print(f"GPU job SUCCESS. Marking {model.id} as COMPLETED.")
                registry[model.id] = {"status": "COMPLETED", "date": TODAY}
            else:
                rc = process.returncode

                if rc == 14:
                    err_name, status_type = "Critical Code Bug", "HALT"
                elif rc == 11:
                    err_name, status_type = "Incompatible Architecture", "MODEL_FAULT"
                elif rc in [12, 137, 9]:
                    err_name, status_type = "Node/GPU Out of Memory", "INFRASTRUCTURE_FAULT"
                elif rc == 10:
                    err_name, status_type = "Data/Path Typo", "INFRASTRUCTURE_FAULT"
                else:
                    err_name, status_type = "Generic Slurm/Job Failure", "INFRASTRUCTURE_FAULT"

                print(f"GPU job FAILED with Exit Code {rc} -> [{err_name}]")

                if status_type == "HALT":
                    print(f"\n[CRITICAL ALERT] Developer Code Bug detected for {model.id}.")
                    print("HALTING ENTIRE PIPELINE TO PREVENT ENDLESS LOOPING.")
                    sys.exit(1)

                elif status_type == "MODEL_FAULT":
                    print(f"Model BLACKLISTED. It will NOT be retried on future runs.")
                    registry[model.id] = {
                        "status": "MODEL_FAULT",
                        "reason": f"{err_name} (Exit {rc})",
                        "date": TODAY,
                    }

                elif status_type == "INFRASTRUCTURE_FAULT":
                    print(f"Flagged for RETRY on next pipeline run.")
                    registry[model.id] = {
                        "status": "INFRASTRUCTURE_FAULT",
                        "reason": f"{err_name} (Exit {rc})",
                        "date": TODAY,
                    }

            save_registry(registry)

            # ---------------------------------------------------------------
            # Cleanup — delete temp images only (masks stay)
            # ---------------------------------------------------------------
            if os.path.exists(temp_data_dir):
                print(f"Cleaning up temp images at {temp_data_dir}...")
                shutil.rmtree(temp_data_dir)

            current_status = registry.get(model.id, {}).get("status")
            if current_status != "INFRASTRUCTURE_FAULT":
                print(f"Cleaning up {model.id} from HF cache...")
                safe_dir_name = "models--" + model.id.replace("/", "--")
                model_path = os.path.join(os.environ.get("HF_HOME", ""), "hub", safe_dir_name)
                if os.path.exists(model_path):
                    shutil.rmtree(model_path)
            else:
                print(f"Keeping {model.id} in cache for retry.")

        except KeyboardInterrupt:
            print("Keyboard interruption, halting.")
            sys.exit(0)

        except Exception as e:
            print(f"Unexpected Pipeline Failure for {model.id}: {e}")
            registry[model.id] = {
                "status": "INFRASTRUCTURE_FAULT",
                "reason": str(e),
                "date": TODAY,
            }
            save_registry(registry)

            temp_data_dir = os.path.join(INPAINT_TMP_ROOT, safe_model_name)
            if os.path.exists(temp_data_dir):
                shutil.rmtree(temp_data_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, nargs="+", default=None,
                        help="Run specific model(s) only, e.g. --model black-forest-labs/FLUX.1-dev")
    parser.add_argument("--staging-dir", default=STAGING_DIR)
    parser.add_argument("--master-prompts-csv", default=MASTER_PROMPTS_CSV)
    parser.add_argument("--masks-dir", default=MASKS_DIR)
    parser.add_argument("--inpaint-tmp-root", default=INPAINT_TMP_ROOT)
    parser.add_argument("--registry-file", default=REGISTRY_FILE)
    parser.add_argument("--inpaint-script", default=INPAINT_SCRIPT)
    parser.add_argument("--inpaint-large-script", default=INPAINT_LARGE_SCRIPT)
    parser.add_argument("--slurm-log-dir", default=SLURM_LOG_DIR)
    args = parser.parse_args()
    REGISTRY_FILE = args.registry_file
    STAGING_DIR = args.staging_dir
    MASTER_PROMPTS_CSV = args.master_prompts_csv
    MASKS_DIR = args.masks_dir
    INPAINT_TMP_ROOT = args.inpaint_tmp_root
    INPAINT_SCRIPT = args.inpaint_script
    INPAINT_LARGE_SCRIPT = args.inpaint_large_script
    SLURM_LOG_DIR = args.slurm_log_dir
    run_inpaint_pipeline(specific_models=args.model)
