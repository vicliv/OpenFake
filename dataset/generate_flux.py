import os
import csv
import torch
from diffusers import FluxPipeline
import sys
import argparse
import numpy as np

# Parse base directory argument for saving outputs and progress
parser = argparse.ArgumentParser(description="Flux image generation with resume support")
parser.add_argument('base_dir', help='Base directory to store outputs and progress')
parser.add_argument('metadata_csv', help='Path to the metadata CSV file')
parser.add_argument('--lora_path', help='Path to a local LoRA weights file (.safetensors)', default=None) # Optional argument for LoRA weights
parser.add_argument('--local_rank', type=int, default=local_rank, help='Local rank for distributed training')
parser.add_argument('--world_size', type=int, default=world_size, help='World size for distributed training')
parser.add_argument('--cache_dir', type=str, default='.cache', help='Cache directory for model weights')
args = parser.parse_args()
base_dir = args.base_dir
lora_path = args.lora_path

# ======== Configuration ========
metadata_csv = args.metadata_csv
output_folder = os.path.join(base_dir, f"gpu_{local_rank}")
max_generated = 100000  # Maximum number of images to generate
batch_size = 4  # Number of images to generate per batch (adjust based on GPU memory)

# Ensure base output directory exists
os.makedirs(os.path.dirname(output_folder), exist_ok=True)
# Create per-GPU output folder
os.makedirs(output_folder, exist_ok=True)

# Progress tracking for resume
progress_path = os.path.join(output_folder, "progress.txt")
if os.path.exists(progress_path):
    with open(progress_path, "r") as pf:
        generated_count = int(pf.read().strip())
else:
    generated_count = 0

# ======== Initialize Flux Pipeline on GPU ========
if torch.cuda.is_available():
    device = f"cuda:0"
    torch.cuda.set_device(0)
else:
    device = "cpu"
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, cache_dir=cache_dir
).to(device)

# ======== Integrate Realism LoRA (local file) ========
if lora_path is not None:
    # Use filename (without extension) as adapter name
    adapter_name = os.path.splitext(os.path.basename(lora_path))[0]
    pipe.load_lora_weights(lora_path)
    # Optionally fuse for faster inference
    pipe.fuse_lora(lora_scale=0.8)
    print(f"Loaded LoRA weights from {lora_path} as '{adapter_name}' adapter.")
else:
    print("No LoRA path provided; running vanilla FluxPipeline.")

# ======== Load Prompts ========
prompts = []
with open(metadata_csv, mode="r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompt = row.get("generated_prompt", "").strip()
        if prompt:
            prompts.append(prompt)
# Distribute prompts across GPUs so each GPU generates a unique subset
if world_size > 1:
    prompts = prompts[local_rank::world_size]
# Skip prompts already generated when resuming
if generated_count > 0:
    prompts = prompts[generated_count:]
# ======== Generate Images in Batches ========
num_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)

sizes = [(1024, 1024), (1024, 512), (512, 1024), (1024, 768), (768, 1024), (1024, 1024), (1152, 768), (768, 1152)]

for i in range(num_batches):
    # Get the current batch of prompts
    start_index = i * batch_size
    end_index = min(start_index + batch_size, len(prompts))
    prompts_batch = prompts[start_index:end_index]
    
    # Randomly select image size for the batch
    size_index = np.random.choice(len(sizes), 1)[0]
    h, w = sizes[size_index]
    
    # Generate images in a batch
    images = pipe(
        prompt=prompts_batch,
        height=h,
        width=w,
        guidance_scale=3.5,
        num_inference_steps=28,
        generator=torch.Generator(device).manual_seed(0)
    ).images

    # Save images
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"image_{generated_count}.png")
        image.save(output_path)
        print(f"Generated image #{generated_count} for prompt: {prompts_batch[i]}")
        generated_count += 1
        # Update resume progress
        with open(progress_path, "w") as pf:
            pf.write(str(generated_count))

print(f"Finished generating {generated_count} images.")