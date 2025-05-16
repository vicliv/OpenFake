from concurrent.futures import ThreadPoolExecutor
import torch
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
import gc
# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Constants ===
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 448
BATCH_SIZE = 1
DEVICE = "cuda:0"
MODEL_NAME = "OpenGVLab/InternVL3-8B"
CACHE_DIR = ".cache"
QUESTION = "Is this image real or AI-generated?"
GENERATION_CONFIG = {
    "do_sample": False,
    "max_new_tokens": 10,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "repetition_penalty": 1.0
}

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image(image_path, transform):
    try:
        image = Image.open(image_path).convert("RGB")
        resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
        pixel_values = transform(resized_image).unsqueeze(0)
        return pixel_values
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
    )
    return model.to(DEVICE).eval(), tokenizer

def predict_batch(model, tokenizer, pixel_batch):
    responses = []
    for pixel_values in pixel_batch:
        try:
            pixel_values = pixel_values.to(dtype=torch.bfloat16, device=DEVICE)
            response = model.chat(tokenizer, pixel_values, QUESTION, GENERATION_CONFIG)
            responses.append(response.strip())
        except Exception as e:
            print(f"Error during prediction: {e}")
            responses.append("error")
    return responses

def run_inference_on_csv(in_csv, output_csv, save_every_n_batches=10):
    df = pd.read_csv(in_csv)
    assert 'filename' in df.columns, "CSV must have a 'filename' column"

    already_done = set()
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        already_done = set(existing_df['filename'].tolist())
        print(f"Found {len(already_done)} already processed images.")
    else:
        existing_df = pd.DataFrame(columns=['filename', 'label', 'prediction'])

    df = df[~df['filename'].isin(already_done)]
    print(f"Processing {len(df)} remaining images...")

    if df.empty:
        print("All images already processed.")
        return

    model, tokenizer = load_model()
    transform = build_transform(IMAGE_SIZE)

    batch_counter = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        pixel = load_image(row['filename'], transform)
        if pixel is None:
            result = (row['filename'], row.get('typ', 'unknown'), "error")
            existing_df = pd.concat([existing_df, pd.DataFrame([result], columns=['filename', 'label', 'prediction'])])
            existing_df.to_csv(output_csv, index=False)
            continue

        # Run prediction safely
        try:
            with torch.no_grad():
                pixel = pixel.to(dtype=torch.bfloat16, device=DEVICE)
                response = model.chat(tokenizer, pixel, QUESTION, GENERATION_CONFIG).strip()
        except Exception as e:
            print(f"Error during prediction: {e}")
            response = "error"

        result = (row['filename'], row.get('typ', 'unknown'), response)
        existing_df = pd.concat([existing_df, pd.DataFrame([result], columns=['filename', 'label', 'prediction'])])
        batch_counter += 1

        # Save progress periodically
        if batch_counter % save_every_n_batches == 0:
            existing_df.to_csv(output_csv, index=False)
            print(f"Saved progress after {batch_counter} images.")

        # Free up memory
        del pixel
        torch.cuda.empty_cache()
        gc.collect()

    # Final save
    existing_df.to_csv(output_csv, index=False)
    print(f"\nDone! Final results saved to: {output_csv}")


if __name__ == "__main__":
    INPUT_CSV = os.path.expanduser("test_image_labels.csv")
    OUTPUT_CSV = "vlm_baseline_output_predictions.csv"
    run_inference_on_csv(INPUT_CSV, OUTPUT_CSV)
