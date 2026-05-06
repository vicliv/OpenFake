import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
import fiftyone as fo
import fiftyone.zoo as foz

load_dotenv()

LLM_CACHE_DIR = "data/models/hub"
DATASET_DIR = os.environ.get("OPEN_IMAGES_DIR", "data/open_images")
EXTRACTED_MASKS_DIR = os.path.join(DATASET_DIR, "extracted_masks")
EXPORT_CSV = os.path.join(DATASET_DIR, "metadata.csv")

# Route FiftyOne to the scratch space to prevent home directory overflow
fo.config.default_dataset_dir = DATASET_DIR
fo.config.dataset_zoo_dir = os.path.join(DATASET_DIR, "zoo")

def download_llm():
    os.makedirs(LLM_CACHE_DIR, exist_ok=True)

    snapshot_download(
            repo_id="google/gemma-3-27b-it",
            cache_dir=LLM_CACHE_DIR,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"], 
            resume_download=True
        )


def download_dataset():
    os.makedirs(EXTRACTED_MASKS_DIR, exist_ok=True)
    
    # This downloads strictly the images containing segmentation masks
    # The max_samples argument prevents downloading the full 2M images initially
    dataset = foz.load_zoo_dataset(
        "open-images-v7",
        split="train",
        label_types=["segmentations"],
        num_workers=4, 
        dataset_name="open-images-masks"
    )
    
    data = []
    
    for sample in dataset:
        if not sample.has_field("segmentations") or sample.segmentations is None:
            continue
            
        detections = sample.segmentations.detections
        if not detections:
            continue
            
        # Select the primary mask for this image
        target_detection = detections
        mask_label = target_detection.label
        
        # Open Images stores masks as boolean arrays. Convert to a standard 0/255 PNG.
        mask_array = target_detection.mask
        if mask_array is None:
            continue
            
        mask_filename = f"{sample.id}_{mask_label.replace(' ', '_')}.png"
        mask_filepath = os.path.join(EXTRACTED_MASKS_DIR, mask_filename)
        
        # Save the mask to the scratch disk
        cv2.imwrite(mask_filepath, (mask_array * 255).astype(np.uint8))
        
        # Open Images lacks paragraph captions, so we construct a base description
        image_description = f"An image containing a {mask_label}"
        
        data.append({
            "image_id": sample.id,
            "image_path": sample.filepath,
            "mask_path": mask_filepath,
            "original_description": image_description,
            "mask_label": mask_label
        })
        
    df = pd.DataFrame(data)
    df.to_csv(EXPORT_CSV, index=False)
    print(f"Extraction complete. Exported {len(df)} records to {EXPORT_CSV}")

if __name__ == "__main__":
    download_llm()
