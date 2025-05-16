import os
import sys
import time
import json
import csv
import shutil
import requests
import pandas as pd
from PIL import Image, ImageFile
from io import BytesIO
import torch
import concurrent.futures
import argparse

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======== Argument Parser ========
parser = argparse.ArgumentParser()
parser.add_argument("--metadata_dir", type=str, default="laion400m/laion400m-met-release/laion400m-meta")
parser.add_argument("--download_dir", type=str, default="downloaded_images")
parser.add_argument("--temp_dir", type=str, default="temp_images")
parser.add_argument("--max_downloads_per_gpu", type=int, default=1000)
parser.add_argument("--max_images_per_gpu", type=int, default=1000000)
parser.add_argument("--prompts_file", type=str, default="prompts.csv")
args = parser.parse_args()

CHECKPOINT_FILE_TEMPLATE = "progress_checkpoint_gpu{}.json"
METADATA_CSV_TEMPLATE = args.prompts_file

# ======== Get GPU/Task ID from Arguments ========
if len(sys.argv) < 2:
    print("Usage: python laion.py <gpu/task id>", flush=True)
    sys.exit(1)
gpu_id = int(sys.argv[1])  # Provided via SLURM_ARRAY_TASK_ID
METADATA_DIR = args.metadata_dir
DOWNLOAD_DIR = args.download_dir
TEMP_DIR = os.path.join(args.temp_dir, f"temp_images_gpu{gpu_id}")
MAX_DOWNLOADS_PER_GPU = args.max_downloads_per_gpu
MAX_IMAGES_PER_GPU = args.max_images_per_gpu
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

# Determine number of tasks in the job array.
num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

# ======== Setup Directories and Files ========
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

metadata_csv = METADATA_CSV_TEMPLATE.format(gpu_id)

if not os.path.exists(metadata_csv):
    with open(metadata_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "generated_prompt", "NSFW", "image_url", "category"])

checkpoint_file = CHECKPOINT_FILE_TEMPLATE.format(gpu_id)

def load_checkpoint():
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    else:
        return {"file_index": 0, "row_index": -1, "download_count": 0}

def save_checkpoint(chkpt):
    with open(checkpoint_file, "w") as f:
        json.dump(chkpt, f)

checkpoint = load_checkpoint()

device = f"cuda"

# ======== Initialize the Model and Processor ========
cache_dir = os.path.join(os.environ.get("SCRATCH", "/tmp"), ".cache")
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map={"": 0},  # Pin entire model to the designated GPU
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )
except Exception as e:
    print(f"Failed to load model: {e}", flush=True)
    sys.exit(1)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=64*28*28, max_pixels=1280*28*28, cache_dir=cache_dir)

def check_image_content(image_input, caption):
    pil_image = Image.open(image_input)
    # skip image if smaller than 64x64
    if pil_image.width < 64 or pil_image.height < 64:
        return {"humans": False, "catastrophes": False, "reasoning": "Image too small."}
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": (
                f"Analyze the provided image and its caption: \"{caption}\". "
                "Provide detailed reasoning on the following two points:\n"
                "1. Does the image contain any real human face(s)? Exclude animations, cartoons, figurines, statues, drawings, paintings, or video games.\n"
                "2. Does the image contain content related to political events, catastrophes, news event, or anything likely to have high emotional impact or polarization? Exclude animations, cartoons, drawings, paintings, or video games.\n\n"
                "Conclude clearly with either 'Humans: yes' or 'Humans: no', and 'Catastrophes: yes' or 'Catastrophes: no'."
            )}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    input_len = inputs.input_ids.shape[-1]

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=200)

    generated_ids_trimmed = [out_ids[input_len:] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    humans = 'humans: yes' in decoded.lower()
    catastrophes = 'catastrophes: yes' in decoded.lower()

    return {"humans": humans, "catastrophes": catastrophes, "reasoning": decoded}

def generate_prompt_from_caption(image_input, caption):
    pil_image = Image.open(image_input)

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": (
                f"Given the image and its caption: \"{caption}\", generate a concise prompt in a single sentence that describes "
                "the image and its format (e.g. photograph, poster, screenshot), including any people present. "
                "Do not mention the caption directly."
            )}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(device)
    input_len = inputs.input_ids.shape[-1]
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids_trimmed = [out_ids[input_len:] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    decoded = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return decoded

def download_image(row, idx):
    caption = row.get("TEXT", "")
    image_url = row.get("URL", "")
    nsfw = row.get("NSFW", "Unknown")
    if not image_url:
        return None
    try:
        response = requests.get(image_url, timeout=0.5)
        if response.status_code != 200:
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"Error downloading or opening image from {image_url}: {e}", flush=True)
        return None

    temp_filename = os.path.join(TEMP_DIR, f"temp_{gpu_id}_{idx}.jpg")
    try:
        image.save(temp_filename)
    except Exception as e:
        print(f"Error saving temporary image: {e}", flush=True)
        return None
    return (temp_filename, caption, image_url, nsfw, idx)

# ======== Main Worker Loop ========
parquet_files = sorted(
    [os.path.join(METADATA_DIR, f) for f in os.listdir(METADATA_DIR) if f.endswith(".parquet")]
)
# Partition files among tasks using the task ID and total number of tasks.
assigned_files = parquet_files[gpu_id::num_tasks]

download_count = checkpoint.get("download_count", 0)

start_time = time.time()
for file_index, file in enumerate(assigned_files):
    if file_index < checkpoint["file_index"]:
        continue

    print(f"Processing file {file_index}: {file}", flush=True)
    try:
        df = pd.read_parquet(file)
    except Exception as e:
        print(f"Error reading {file}: {e}", flush=True)
        continue

    start_row = checkpoint["row_index"] + 1 if file_index == checkpoint["file_index"] else 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        buffer = []
        for idx, row in df.iterrows():
            if idx < start_row:
                continue
            if download_count >= MAX_IMAGES_PER_GPU:
                break

            # Submit download task
            future = executor.submit(download_image, row, idx)
            buffer.append((future, idx))

            # If buffer is full, process the buffered downloads
            if len(buffer) >= 10:
                for future, buffered_idx in buffer:
                    result = future.result()
                    if result is None:
                        continue
                    (temp_filename, caption, image_url, nsfw, r_idx) = result
                    try:
                        image_content = check_image_content(temp_filename, caption)
                        if image_content["humans"] or image_content["catastrophes"]:
                            generated_prompt = generate_prompt_from_caption(temp_filename, caption)
                            category = "both" if image_content["humans"] and image_content["catastrophes"] else "human" if image_content["humans"] else "catastrophe"
                            reasoning = image_content["reasoning"]
                        else:
                            os.remove(temp_filename)
                            continue
                        if generated_prompt.startswith("Prompt:"):
                            generated_prompt = generated_prompt.replace("Prompt:", "", 1).strip()
                        for phrase in ["The image is ", "The image shows ", "The image depicts ", 
                                       "The image contains ", "The image features ", "The image portrays ",
                                       "The image includes ", "The image presents ", "The image exhibits "]:
                            if generated_prompt.startswith(phrase):
                                generated_prompt = generated_prompt.replace(phrase, "", 1)
                                generated_prompt = generated_prompt[0].upper() + generated_prompt[1:]
                                break
                        if download_count < MAX_DOWNLOADS_PER_GPU:
                            final_filename = f"gpu{gpu_id}_image_{download_count}.jpg"
                            final_path = os.path.join(DOWNLOAD_DIR, final_filename)
                            shutil.move(temp_filename, final_path)
                        else:
                            # Remove temp file when not saving to download directory
                            os.remove(temp_filename)
                        
                        with open(metadata_csv, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow([caption, generated_prompt, nsfw, image_url, category])
                        
                        download_count += 1
                        print(f"Downloaded image #{download_count} ({category})", flush=True)

                    except Exception as e:
                        print(f"Error during processing: {e}", flush=True)
                        if os.path.exists(temp_filename):
                            os.remove(temp_filename)

                    checkpoint["file_index"] = file_index
                    checkpoint["row_index"] = buffered_idx
                    checkpoint["download_count"] = download_count
                    save_checkpoint(checkpoint)
                buffer = []

        # Process any remaining tasks in the buffer
        for future, buffered_idx in buffer:
            result = future.result()
            if result is None:
                continue
            (temp_filename, caption, image_url, nsfw, r_idx) = result
            try:
                image_content = check_image_content(temp_filename, caption)
                if image_content["humans"] or image_content["catastrophes"]:
                    generated_prompt = generate_prompt_from_caption(temp_filename, caption)
                    category = "both" if image_content["humans"] and image_content["catastrophes"] else "human" if image_content["humans"] else "catastrophe"
                    reasoning = image_content["reasoning"]
                else:
                    os.remove(temp_filename)
                    continue
                if generated_prompt.startswith("Prompt:"):
                    generated_prompt = generated_prompt.replace("Prompt:", "", 1).strip()
                for phrase in ["The image is ", "The image shows ", "The image depicts ", 
                               "The image contains ", "The image features ", "The image portrays ",
                               "The image includes ", "The image presents ", "The image exhibits "]:
                    if generated_prompt.startswith(phrase):
                        generated_prompt = generated_prompt.replace(phrase, "", 1)
                        generated_prompt = generated_prompt[0].upper() + generated_prompt[1:]
                        break
                if download_count < MAX_DOWNLOADS_PER_GPU:
                    final_filename = f"gpu{gpu_id}_image_{download_count}.jpg"
                    final_path = os.path.join(DOWNLOAD_DIR, final_filename)
                    shutil.move(temp_filename, final_path)
                else:
                    # Remove temp file when not saving to download directory
                    os.remove(temp_filename)
                
                with open(metadata_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([caption, generated_prompt, nsfw, image_url, category])

                download_count += 1
                print(f"Downloaded image #{download_count} ({category})", flush=True)
            except Exception as e:
                print(f"Error during processing: {e}", flush=True)
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            checkpoint["file_index"] = file_index
            checkpoint["row_index"] = buffered_idx
            checkpoint["download_count"] = download_count
            save_checkpoint(checkpoint)

    checkpoint["file_index"] = file_index + 1
    checkpoint["row_index"] = -1
    checkpoint["download_count"] = download_count
    save_checkpoint(checkpoint)

    if download_count >= MAX_IMAGES_PER_GPU:
        break

elapsed = int(time.time() - start_time)
print(f"Finished. Downloaded {download_count} images in {elapsed} seconds.", flush=True)