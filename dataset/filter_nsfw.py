"""
Using nsfw model (NSFW_MODEL_PATH), goes through the REDDIT_IMAGES_DIR and deletes all NSFW images
which have a score greater than the NSFW_DETECTION_THRESHOLD. 

Started from `start_reddit_pipeline.sh` automatically, but can be run manually.

USAGE: 
bash start_reddit_pipeline.sh REDDIT_IMAGES_DIR
"""

import os
import sys
import argparse
from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument("reddit_images_dir")
parser.add_argument("--model-path", default="data/models/nsfw_filtering_model")
parser.add_argument("--threshold", type=float, default=0.7)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

NSFW_MODEL_PATH = args.model_path
REDDIT_IMAGES_DIR = args.reddit_images_dir
NSFW_DETECTION_THRESHOLD = args.threshold
NSFW_DETECTION_BATCH_SIZE = args.batch_size

print("Loading local NSFW detection model...")
try:
    # device=0 ensures it targets the allocated GPU
    nsfw_classifier = pipeline("image-classification", model=NSFW_MODEL_PATH, device=0) 
except Exception as e:
    print(f"Failed to load local model. Ensure the path is correct: {e}")
    sys.exit(1)

if isinstance(REDDIT_IMAGES_DIR, list):
    print(REDDIT_IMAGES_DIR)
    sys.exit(1)

# Grab all valid file paths into a standard list
filepaths = [
    os.path.join(REDDIT_IMAGES_DIR, f) for f in os.listdir(REDDIT_IMAGES_DIR) 
    if os.path.isfile(os.path.join(REDDIT_IMAGES_DIR, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

print(f"Found {len(filepaths)} images. Beginning batched inference...")

# 2. Create a native Python generator
def data_generator(paths):
    for path in paths:
        yield path

try:
    # Pass the generator directly to the pipeline
    for filepath, result in zip(filepaths, nsfw_classifier(data_generator(filepaths), batch_size=NSFW_DETECTION_BATCH_SIZE)):
        
        is_nsfw = any(r['label'] == 'nsfw' and r['score'] >= NSFW_DETECTION_THRESHOLD for r in result)
        
        if is_nsfw:
            print(f"Removing NSFW image: {os.path.basename(filepath)}")
            os.remove(filepath)

    print("Successfully filtered NSFW images")
except Exception as e:
    print(f"Filtering failed: {e}")
    sys.exit(1)
