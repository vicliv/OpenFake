"""
One-time mask expansion script.
Processes all masks in MASKS_DIR in-place, applying randomized dilation
so the inpainting model has breathing room and the detection model
can't learn a single mask shape.

Each mask gets:
- Random dilation: 3-10% of shorter image dimension
- Random kernel: ellipse, rectangle, or cross
- Random edge treatment: sharp, slight blur, or moderate blur

Saves a marker file so it won't re-process if run again.

Usage:
    python expand_masks.py                    # process all
    python expand_masks.py --preview 10       # preview 10 random masks without saving
    python expand_masks.py --workers 8        # parallel processing
"""

import argparse
import os
import random
import glob
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

MASKS_DIR = "data/open_images/masks"
MARKER_FILE = os.path.join(MASKS_DIR, ".expanded")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=int, default=0,
                        help="Preview N random masks (saves before/after PNGs to preview/ dir)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if already done")
    parser.add_argument("--skip", type=int, default=0,
                        help="Skip the first N masks (for resuming)")
    parser.add_argument("--masks-dir", default=MASKS_DIR)
    return parser.parse_args()


def expand_single_mask(mask_path):
    """
    Expand a single mask with randomized parameters.
    Returns True on success, False on failure.
    """
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return False, mask_path, "Could not read"

        h, w = mask.shape
        shorter_dim = min(h, w)

        # Random dilation: 3-10% of shorter dimension
        # Minimum 5px, maximum 80px
        dilation_pct = random.uniform(0.03, 0.10)
        dilation_px = int(shorter_dim * dilation_pct)
        dilation_px = max(5, min(dilation_px, 80))

        # Random kernel shape
        kernel_type = random.choice([
            cv2.MORPH_ELLIPSE,
            cv2.MORPH_RECT,
            cv2.MORPH_CROSS,
        ])

        # Slight randomness in kernel aspect ratio (not always square)
        kw = dilation_px * 2 + 1
        kh_offset = random.randint(-3, 3)
        kh = max(3, kw + kh_offset)

        kernel = cv2.getStructuringElement(kernel_type, (kw, kh))

        # Random number of iterations (usually 1, occasionally 2 for extra padding)
        iterations = random.choices([1, 2], weights=[0.8, 0.2])[0]

        dilated = cv2.dilate(mask, kernel, iterations=iterations)

        # Random edge treatment
        edge_type = random.choices(
            ["sharp", "slight_blur", "moderate_blur"],
            weights=[0.4, 0.4, 0.2]
        )[0]

        if edge_type == "slight_blur":
            blur_size = random.choice([3, 5])
            dilated = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
            _, dilated = cv2.threshold(dilated, 127, 255, cv2.THRESH_BINARY)
        elif edge_type == "moderate_blur":
            blur_size = random.choice([5, 7, 9])
            dilated = cv2.GaussianBlur(dilated, (blur_size, blur_size), 0)
            _, dilated = cv2.threshold(dilated, 100, 255, cv2.THRESH_BINARY)

        # Overwrite the original mask
        cv2.imwrite(mask_path, dilated)
        return True, mask_path, f"dil={dilation_px}px kernel={kernel_type} iter={iterations} edge={edge_type}"

    except Exception as e:
        return False, mask_path, str(e)


def preview_masks(n):
    """Show before/after for N random masks without modifying originals."""
    os.makedirs("preview", exist_ok=True)
    all_masks = glob.glob(os.path.join(MASKS_DIR, "*.png"))
    samples = random.sample(all_masks, min(n, len(all_masks)))

    for mask_path in samples:
        fname = os.path.basename(mask_path)

        # Read original
        original = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            continue

        # Save original
        cv2.imwrite(f"preview/{fname}_before.png", original)

        # Process a copy
        h, w = original.shape
        shorter_dim = min(h, w)
        dilation_px = int(shorter_dim * random.uniform(0.03, 0.10))
        dilation_px = max(5, min(dilation_px, 80))

        kernel_type = random.choice([cv2.MORPH_ELLIPSE, cv2.MORPH_RECT, cv2.MORPH_CROSS])
        kw = dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(kernel_type, (kw, kw))
        dilated = cv2.dilate(original, kernel, iterations=1)

        cv2.imwrite(f"preview/{fname}_after.png", dilated)
        print(f"  {fname}: {original.shape} dilation={dilation_px}px", flush=True)

    print(f"\nSaved {len(samples)} before/after pairs to preview/", flush=True)


def main():
    args = parse_args()
    global MASKS_DIR, MARKER_FILE
    MASKS_DIR = args.masks_dir
    MARKER_FILE = os.path.join(MASKS_DIR, ".expanded")

    if args.preview > 0:
        preview_masks(args.preview)
        return

    # Check if fully done
    if os.path.exists(MARKER_FILE) and not args.force:
        print(f"Masks already expanded (marker file exists at {MARKER_FILE}).", flush=True)
        print(f"Use --force to re-process.", flush=True)
        return

    all_masks = sorted(glob.glob(os.path.join(MASKS_DIR, "*.png")))
    print(f"Found {len(all_masks)} masks total.", flush=True)

    if not all_masks:
        print("No masks found.", flush=True)
        return

    # Skip already-processed masks
    if args.skip > 0:
        all_masks = all_masks[args.skip:]
        print(f"Skipping first {args.skip}, processing {len(all_masks)} remaining.", flush=True)

    if not all_masks:
        print("Nothing to process after skip.", flush=True)
        return

    # Process in batches
    success = 0
    failed = 0
    done = 0
    batch_size = 1000

    for batch_start in range(0, len(all_masks), batch_size):
        batch = all_masks[batch_start:batch_start + batch_size]

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(expand_single_mask, batch)

            for ok, path, info in results:
                done += 1
                if ok:
                    success += 1
                else:
                    failed += 1
                    print(f"  FAILED: {os.path.basename(path)}: {info}", flush=True)

        print(f"  Processed {done}/{len(all_masks)} ({success} ok, {failed} failed)...", flush=True)

    print(f"\nDone. {success} expanded, {failed} failed out of {len(all_masks)}.", flush=True)


if __name__ == "__main__":
    main()
