"""
Quick viewer: picks a random row from a prompt shard CSV,
downloads the image from Open Images, loads the mask from local storage,
and shows them side by side with the prompt.

Usage:
    python view_prompt_sample.py                          # random from first shard
    python view_prompt_sample.py --shard 3                # random from shard 3
    python view_prompt_sample.py --shard 0 --index 42     # specific row
"""

import argparse
import os
import random
import csv
import subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import textwrap

SHARD_DIR = "data/open_images/prompt_shards"
LOCAL_MASKS_DIR = "data/open_images/masks"

# Open Images image URL
IMAGE_BASE = "https://s3.amazonaws.com/open-images-dataset/train"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default="samples")
    parser.add_argument("--count", type=int, default=1, help="Number of samples to grab")
    parser.add_argument("--shard-dir", default=SHARD_DIR)
    parser.add_argument("--masks-dir", default=LOCAL_MASKS_DIR)
    return parser.parse_args()


def load_shard(shard_index):
    path = os.path.join(SHARD_DIR, f"shard_{shard_index}.csv")
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def download_file(url, dest):
    result = subprocess.run(
        ["curl", "-sS", "-L", "-f", "-o", dest, "--max-time", "30", url],
        capture_output=True, text=True
    )
    return result.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0


def find_local_mask(mask_path):
    """Check local masks directory for the mask file."""
    # Masks are organized in subdirectories by first char of ImageID
    first_char = mask_path[0].lower()

    # Try a few possible local paths
    candidates = [
        os.path.join(LOCAL_MASKS_DIR, mask_path),
        os.path.join(LOCAL_MASKS_DIR, first_char, mask_path),
        os.path.join(LOCAL_MASKS_DIR, f"train-masks-{first_char}", mask_path),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def main():
    args = parse_args()
    global SHARD_DIR, LOCAL_MASKS_DIR
    SHARD_DIR = args.shard_dir
    LOCAL_MASKS_DIR = args.masks_dir
    rows = load_shard(args.shard)
    print(f"Loaded shard {args.shard}: {len(rows)} rows", flush=True)

    os.makedirs(args.save_dir, exist_ok=True)

    for sample_num in range(args.count):
        if args.index is not None:
            idx = args.index
        else:
            idx = random.randint(0, len(rows) - 1)

        row = rows[idx]
        image_id = row["ImageID"]
        mask_path = row["MaskPath"]
        class_name = row["ClassName"]
        prompt = row["generated_prompt"]

        print(f"\n{'='*60}")
        print(f"Sample {sample_num+1} — Row {idx}")
        print(f"ImageID:   {image_id}")
        print(f"MaskPath:  {mask_path}")
        print(f"Class:     {class_name}")
        print(f"Prompt:    {prompt[:150]}...")

        # Download image
        image_url = f"{IMAGE_BASE}/{image_id}.jpg"
        image_dest = os.path.join(args.save_dir, f"{image_id}.jpg")

        if not os.path.exists(image_dest):
            print(f"Downloading image: {image_url}")
            img_ok = download_file(image_url, image_dest)
            print(f"  -> {'OK' if img_ok else 'FAILED'}")
        else:
            img_ok = True
            print(f"Image already cached.")

        if not img_ok:
            print("Could not download image, skipping.")
            continue

        # Find mask locally
        mask_local = find_local_mask(mask_path)
        mask_ok = mask_local is not None

        if mask_ok:
            print(f"Found local mask: {mask_local}")
        else:
            print(f"Mask not found locally. Available mask dirs:")
            if os.path.exists(LOCAL_MASKS_DIR):
                print(f"  {os.listdir(LOCAL_MASKS_DIR)[:10]}")
            else:
                print(f"  Masks directory not found at {LOCAL_MASKS_DIR}")

        # Build visualization — images on top, prompt text below
        n_cols = 2 if mask_ok else 1
        fig = plt.figure(figsize=(7 * n_cols, 9))

        # Top row: image(s)
        ax_img = fig.add_axes([0.02, 0.30, 0.46 if mask_ok else 0.96, 0.65])
        img = Image.open(image_dest)
        ax_img.imshow(img)
        ax_img.set_title(f"Image: {image_id}", fontsize=10)
        ax_img.axis("off")

        if mask_ok:
            ax_mask = fig.add_axes([0.52, 0.30, 0.46, 0.65])
            mask = Image.open(mask_local)
            ax_mask.imshow(mask, cmap="gray")
            ax_mask.set_title(f"Mask: {class_name}", fontsize=10)
            ax_mask.axis("off")

        # Bottom: prompt text in its own area
        wrapped = textwrap.fill(prompt, width=100)
        fig.text(0.5, 0.18, f"[{class_name}] Prompt:", ha="center", va="top",
                 fontsize=10, fontweight="bold", transform=fig.transFigure)
        fig.text(0.5, 0.15, wrapped, ha="center", va="top",
                 fontsize=8, transform=fig.transFigure,
                 family="monospace")

        out_path = os.path.join(args.save_dir, f"sample_{image_id}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close()


if __name__ == "__main__":
    main()
