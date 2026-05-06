import os
import sys
import argparse
import torch
import gc
import traceback
from diffusers import DiffusionPipeline # Notice we use the universal pipeline here
from prompt_manager import CSVPromptStreamer
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("model_id")
parser.add_argument("total_target_count", type=int)
parser.add_argument("model_type")
parser.add_argument("base_model_id")
parser.add_argument("--staging-dir", default="data/staging_images")
parser.add_argument("--prompts-csv", default="data/prompts/unused_prompts2.csv")
args = parser.parse_args()

model_id = args.model_id
total_target_count = args.total_target_count
model_type = args.model_type
base_model_id = args.base_model_id

task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
task_count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

videos_per_task = total_target_count // task_count
start_index = task_id * videos_per_task
end_index = start_index + videos_per_task

if task_id == task_count - 1:
    end_index = total_target_count

target_count_for_this_gpu = end_index - start_index
staging_dir = args.staging_dir
os.makedirs(staging_dir, exist_ok=True)

prompt_generator = CSVPromptStreamer(args.prompts_csv)
for _ in range(start_index):
    prompt_generator.get_next_prompt()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

EXIT_DATA_FAULT = 10
EXIT_MODEL_FAULT = 11
EXIT_MEMORY_FAULT = 12
EXIT_CODE_BUG = 14

try:
    print(f"Task {task_id}: Loading standalone VIDEO model ({model_id})...")
    # DiffusionPipeline automatically routes to TextToVideoSDPipeline, CogVideoX, etc.
    pipeline = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        use_safetensors=True,
        local_files_only=True
    )
    
    if hasattr(pipeline, "enable_model_cpu_offload"):
        pipeline.enable_model_cpu_offload() 
        print(f"Task {task_id}: CPU offloading enabled.", flush=True)
    else:
        pipeline = pipeline.to(device)

    # VAE Slicing is absolutely critical for video. It decodes the frames 1 by 1 instead of all 16 at once.
    if hasattr(pipeline, "enable_vae_slicing"):
        pipeline.enable_vae_slicing()
        print(f"Task {task_id}: VAE slicing enabled.", flush=True)
    if hasattr(pipeline, "enable_vae_tiling"):
        pipeline.enable_vae_tiling()

    # Video is massive. Start with a tiny batch size.
    optimal_batch_size = 2 
    videos_generated = 0
    pending_prompts = []    
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    while videos_generated < target_count_for_this_gpu:
        current_chunk_size = min(optimal_batch_size, target_count_for_this_gpu - videos_generated)
        
        while len(pending_prompts) < current_chunk_size:
            pending_prompts.append(prompt_generator.get_next_prompt())
            
        attempt_prompts = pending_prompts[:current_chunk_size]
        
        try:
            # Most video pipelines respect the num_frames argument.
            # We request 16 frames as a standard baseline.
            kwargs = {"prompt": attempt_prompts}
            if "num_frames" in pipeline.__call__.__code__.co_varnames:
                kwargs["num_frames"] = 16

            results = pipeline(**kwargs)
            
            # results.frames is usually a List of Lists of PIL Images
            # Shape: [batch_size, num_frames, height, width, channels]
            for v_idx, video_frames in enumerate(results.frames):
                
                # Check for the NaN bug on the first frame of the video
                if hasattr(video_frames, "getbbox") and not video_frames.getbbox():
                    raise ArithmeticError("BlackImage_NaN")
                
                # --- FRAME EXTRACTION LOGIC ---
                # We want exactly 10 frames per video, evenly spaced
                total_generated_frames = len(video_frames)
                step_size = max(1, total_generated_frames / 10.0)
                
                # Extract indices: e.g., 0, 1.6, 3.2 -> [0, 1, 3, 4, ...]
                frame_indices_to_save = [int(i * step_size) for i in range(10) if int(i * step_size) < total_generated_frames]
                
                global_index = start_index + videos_generated + v_idx
                safe_model_name = model_id.replace("/", "_")
                
                for f_idx in frame_indices_to_save:
                    frame = video_frames[f_idx]
                    filename = f"hf_vid_{safe_model_name}_{today_date}_{global_index}_frame{f_idx}.png"
                    frame.save(os.path.join(staging_dir, filename))
                
            videos_generated += current_chunk_size
            pending_prompts = [] 
            
            print(f"Task {task_id} generated {videos_generated}/{target_count_for_this_gpu} videos... (Batch Size: {optimal_batch_size})", flush=True)

        except ArithmeticError as e:
            if "BlackImage_NaN" in str(e):
                print(f"Task {task_id}: NaN Video. Auto-healing VAE to FP32...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                
                if hasattr(pipeline, "upcast_vae"):
                    pipeline.upcast_vae()
                elif hasattr(pipeline, "vae"):
                    pipeline.vae.to(dtype=torch.float32)
                else:
                    sys.exit(EXIT_MODEL_FAULT)
                continue

        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            torch.cuda.empty_cache()
            if optimal_batch_size == 1:
                sys.exit(EXIT_MEMORY_FAULT)
            optimal_batch_size = max(1, optimal_batch_size // 2)
            print(f"Task {task_id}: Halved batch size to {optimal_batch_size}...", flush=True)

        except Exception as inner_e:
            if "cuda out of memory" in str(inner_e).lower() or "oom" in str(inner_e).lower():
                gc.collect()
                torch.cuda.empty_cache()
                if optimal_batch_size == 1:
                    sys.exit(EXIT_MEMORY_FAULT)
                optimal_batch_size = max(1, optimal_batch_size // 2)
            else:
                raise inner_e 

    print(f"Success! Task {task_id} finished generating videos for {model_id}", flush=True)
    
except Exception as e:
    error_msg = str(e).lower()
    
    if "cuda out of memory" in error_msg:
        sys.exit(EXIT_MEMORY_FAULT)
    elif isinstance(e, FileNotFoundError) or isinstance(e, OSError) or "no such file or directory" in error_msg:
        hf_keywords = ["model_index.json", "config.json", "scheduler", "huggingface", "safetensors"]
        if any(kw in error_msg for kw in hf_keywords):
            sys.exit(EXIT_MODEL_FAULT) 
        else:
            sys.exit(EXIT_DATA_FAULT) 
    elif any(kw in error_msg for kw in ["weight", "pipeline", "expected str", "time embedding", "incorrect config", "dimension", "black image"]):
        sys.exit(EXIT_MODEL_FAULT) 
    else:
        print(f"Task {task_id} CRITICAL CODE BUG:\n{traceback.format_exc()}", flush=True)
        sys.exit(EXIT_CODE_BUG)
