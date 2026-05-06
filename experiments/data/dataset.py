from __future__ import annotations

import random
from typing import Callable, Literal, List, Optional, Tuple

import torch
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True

_MAX_RETRIES = 10
_MIN_IMAGE_EDGE = 2
_MAX_IMAGE_SIDE = 2048  # cap before decode to avoid 400 MB+ bitmaps per worker


def _load_pil(path: str) -> Image.Image:
    """Load an image as RGB, handling palette PNGs with transparency."""
    with open(path, "rb") as f:
        img = Image.open(f)
        # For JPEG, hint the decoder to subsample before decompressing.
        # draft() is a no-op for non-JPEG formats; thumbnail() covers those.
        if max(img.size) > _MAX_IMAGE_SIDE:
            img.draft(None, (_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))
        img.load()
    if max(img.size) > _MAX_IMAGE_SIDE:
        img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE), Image.LANCZOS)
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA").convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    if min(img.size) < _MIN_IMAGE_EDGE:
        raise ValueError(f"image too small after decode: size={img.size}")
    return img


class OpenFakeDataset(torch.utils.data.Dataset):
    """Dataset that loads images from a pre-resolved list of (path, label) pairs.

    Workers return ``{"image": PIL.Image, "labels": int}`` — the processor
    (resize + normalize → tensor) is applied by the collator in the main
    process, which reduces IPC payload from ~800 KB/image (float32 tensor) to
    ~200 KB/image (uint8 PIL) and frees workers to focus on I/O + augmentation.

    Args:
        samples:    List of ``(absolute_path, label)`` or
                    ``(absolute_path, label, model_name)`` where label is
                    0 (real) or 1 (fake).
        transform:  Optional torchvision transform applied to the PIL image.
                    Pass ``None`` for eval.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int] | Tuple[str, int, str]],
        transform: Optional[Callable] = None,
        on_error: Literal["resample", "blank"] = "resample",
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.on_error = on_error
        self._corrupted: set = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        last_err = None
        for attempt in range(_MAX_RETRIES):
            sample_idx = idx if attempt == 0 else random.randrange(len(self.samples))
            if sample_idx in self._corrupted:
                continue

            sample = self.samples[sample_idx]
            path, label = sample[0], sample[1]
            model_name = sample[2] if len(sample) > 2 else ""
            try:
                img = _load_pil(path)
            except (UnidentifiedImageError, OSError, ValueError) as e:
                last_err = e
                print(f"[WARN] Skipping unreadable image: {path} ({e})")
                self._corrupted.add(sample_idx)
                if self.on_error == "blank":
                    img = Image.new("RGB", (384, 384))
                    return {"image": img, "labels": label, "model_name": model_name}
                continue

            if self.transform:
                try:
                    img = self.transform(img)
                except Exception as e:
                    last_err = e
                    print(f"[WARN] Skipping transform-failing image: {path} ({e})")
                    self._corrupted.add(sample_idx)
                    continue

            return {"image": img, "labels": label, "model_name": model_name}

        raise RuntimeError(
            f"Failed to load a valid image after {_MAX_RETRIES} attempts. "
            f"Last error: {last_err}"
        )
