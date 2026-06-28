"""
Generates TOTAL_AMT_IMAGES_TO_GENERATE of MODEL_ID
If it's a LoRA model, the BASE_MODEL_ID is used instead with the weights
Images are stored in STAGING_DIR, metadata appended to unified metadata.csv

Usage:
Called automatically from hugging_face_pipeline.py --> submit_generate_batch_samples.sh but can be called directly as 

python generate_batch_samples.py MODEL_ID TOTAL_AMT_IMAGES_TO_GENERATE MODEL_TYPE BASE_MODEL_ID RELEASE_DATE
"""

import os
import csv
import fcntl
import argparse
import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv
from datetime import datetime
import gc
import traceback

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("total_images", type=int)
parser.add_argument("model_type")
parser.add_argument("base_model_id")
parser.add_argument("release_date")
parser.add_argument("--staging-dir", default="data/staging_images")
parser.add_argument("--prompts-csv", default="data/prompts/unused_prompts2.csv")
args = parser.parse_args()

STAGING_DIR = args.staging_dir
MODEL_ID = args.model_id
TOTAL_AMT_IMAGES_TO_GENERATE = args.total_images
MODEL_TYPE = args.model_type
BASE_MODEL_ID = args.base_model_id
RELEASE_DATE = args.release_date

METADATA_FIELDS = ["filename", "prompt", "label", "model", "type", "release_date", "packaged"]
safe_model_name = MODEL_ID.replace("/", "_")
METADATA_CSV = os.path.join(STAGING_DIR, "metadata.csv")

# Error codes
EXIT_DATA_FAULT = 10
EXIT_MODEL_FAULT = 11
EXIT_MEMORY_FAULT = 12
EXIT_CODE_BUG = 14

# Array Logic: Determine this GPU's workload
task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

images_per_task = TOTAL_AMT_IMAGES_TO_GENERATE // task_count
start_index = task_id * images_per_task
end_index = start_index + images_per_task

# Ensure the final task picks up any remainder due to uneven division
if task_id == task_count - 1:
    end_index = TOTAL_AMT_IMAGES_TO_GENERATE

target_count_for_this_gpu = end_index - start_index

os.makedirs(STAGING_DIR, exist_ok=True)

prompt_generator = CSVPromptStreamer(args.prompts_csv)

# Fast-forward the prompt generator so it doesn't overlap with other GPUs
for _ in range(start_index):
    prompt_generator.get_next_prompt()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32


def append_metadata_rows(rows):
    """Append metadata rows to the unified metadata CSV with file locking for array task safety."""
    file_exists = os.path.exists(METADATA_CSV) and os.path.getsize(METADATA_CSV) > 0
    with open(METADATA_CSV, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


try:
    if MODEL_TYPE == "LoRA":
        print(f"Task {task_id}: Loading Base Model ({BASE_MODEL_ID}) for LoRA injection...")
        pipeline = AutoPipelineForText2Image.from_pretrained(
            BASE_MODEL_ID, 
            torch_dtype=torch_dtype,
            use_safetensors=True,
            requires_safety_checker=False,
            local_files_only=True
        )
        print(f"Task {task_id}: Injecting LoRA weights from {MODEL_ID}...")
        pipeline.load_lora_weights(MODEL_ID, local_files_only=True)
    else:
        print(f"Task {task_id}: Loading standalone model ({MODEL_ID})...")
        try:
            pipeline = AutoPipelineForText2Image.from_pretrained(
                MODEL_ID, 
                torch_dtype=torch_dtype,
                use_safetensors=True,
                requires_safety_checker=False,
                local_files_only=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"AutoPipeline failed: {e}. Trying DiffusionPipeline...", flush=True)
            pipeline = DiffusionPipeline.from_pretrained(
                MODEL_ID, torch_dtype=torch_dtype,
                use_safetensors=True, local_files_only=True,
                trust_remote_code=True
            )
            
    #Params to reduce GPU/CPU usage 
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload() 
        print(f"Task {task_id}: CPU offloading enabled. (Safely handling VRAM)", flush=True)
    else:
        pipeline = pipeline.to(device)
        print(f"Task {task_id}: Model manually moved to GPU.", flush=True)

    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print(f"Task {task_id}: VAE slicing enabled.", flush=True)

    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()
        print(f"Task {task_id}: VAE tiling enabled.", flush=True)
    
    optimal_batch_size = 12 # Start greedy, half it every time cuda out of memory
    images_generated = 0
    pending_prompts = []    # Cache prompts so we don't lose them if a batch fails
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    while images_generated < target_count_for_this_gpu:
        current_chunk_size = min(optimal_batch_size, target_count_for_this_gpu - images_generated)
        
        # Buffer prompts so we don't skip them if the batch fails
        while len(pending_prompts) < current_chunk_size:
            pending_prompts.append(prompt_generator.get_next_prompt())
            
        attempt_prompts = pending_prompts[:current_chunk_size]
        
        try:
            results = pipeline(prompt=attempt_prompts)
            
            metadata_batch = []
            for j, img in enumerate(results.images):
                # global_index ensures files are numbered correctly from 0 to 9999 across all nodes
                global_index = start_index + images_generated + j
                filename = f"hf_{safe_model_name}_{today_date}_{global_index}.png"
                img.save(os.path.join(STAGING_DIR, filename))
                
                metadata_batch.append({
                    "filename": filename,
                    "prompt": attempt_prompts[j],
                    "label": "fake",
                    "model": MODEL_ID,
                    "type": MODEL_TYPE,
                    "release_date": RELEASE_DATE,
                    "packaged": False,
                })
            
            # Write all metadata for this batch in one locked operation
            append_metadata_rows(metadata_batch)
                
            images_generated += current_chunk_size
            pending_prompts = [] # Clear cached prompts on success
            
            if images_generated % 100 < optimal_batch_size and images_generated > 0:
                print(f"Task {task_id} generated {images_generated}/{target_count_for_this_gpu}... (Locked Batch Size: {optimal_batch_size})", flush=True)

        # Catch PyTorch OOMs *inside* the loop so we can retry instead of crashing
        except torch.cuda.OutOfMemoryError as e:
            print(f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing memory...", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            
            if optimal_batch_size == 1:
                print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                sys.exit(EXIT_MEMORY_FAULT)
                
            optimal_batch_size = max(1, optimal_batch_size // 2)
            print(f"Task {task_id}: Halved batch size. Retrying with {optimal_batch_size}...", flush=True)

        except Exception as inner_e:
            # Catch sneaky generic string OOMs, otherwise escalate to the outer try/except block
            if "cuda out of memory" in str(inner_e).lower() or "oom" in str(inner_e).lower():
                print(f"Task {task_id} VRAM Overload at batch {optimal_batch_size}. Flushing memory...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                
                if optimal_batch_size == 1:
                    print(f"Task {task_id} FATAL: Model too large for 1 image.", flush=True)
                    sys.exit(EXIT_MEMORY_FAULT)
                    
                optimal_batch_size = max(1, optimal_batch_size // 2)
                print(f"Task {task_id}: Halved batch size. Retrying with {optimal_batch_size}...", flush=True)
            else:
                # Escalates things like Architecture Errors to the outer catch
                raise inner_e 

    print(f"Success! Task {task_id} finished generating for {MODEL_ID}", flush=True)
    
except Exception as e:
    error_msg = str(e)
    error_msg_lower = error_msg.lower()
    
    if "cuda out of memory" in error_msg_lower:
        print(f"Task {task_id} VRAM OOM Error during init: {e}", flush=True)
        sys.exit(EXIT_MEMORY_FAULT)
        
    elif isinstance(e, FileNotFoundError) or isinstance(e, OSError) or "no such file or directory" in error_msg_lower:
        hf_keywords = ["model_index.json", "config.json", "scheduler", "huggingface", "safetensors"]
        
        if any(keyword in error_msg_lower for keyword in hf_keywords):
            print(f"Task {task_id} Model Fault: Missing diffusers config/architecture: {e}", flush=True)
            sys.exit(EXIT_MODEL_FAULT) 
        else:
            print(f"Task {task_id} Data/Path Error (Likely a typo in your paths): {e}", flush=True)
            sys.exit(EXIT_DATA_FAULT) 

    elif any(kw in error_msg_lower for kw in ["weight", "pipeline", "expected str", "time embedding", "incorrect config", "dimension"]):
        print(f"Task {task_id} Model Architecture Error: {e}", flush=True)
        sys.exit(EXIT_MODEL_FAULT) 
        
    else:
        print(f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True)
        sys.exit(EXIT_CODE_BUG)
