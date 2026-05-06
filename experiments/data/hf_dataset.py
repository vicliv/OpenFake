from __future__ import annotations

from typing import Callable, Optional

import glob
import os

import torch
from PIL import Image
import datasets
from datasets import load_dataset


class HFMapDataset(torch.utils.data.Dataset):
    """Map-style wrapper around a locally-cached HuggingFace dataset.

    Pass ``data_dir`` pointing to the snapshot's data/ folder (where the
    train-*.parquet files live) to bypass the hub download machinery entirely
    — no lock files are created on the hub cache, so scratch quota is not hit.
    The datasets builder cache is redirected to SLURM_TMPDIR so the small
    builder lock file doesn't land on quota-constrained scratch either.
    Arrow caching is disabled so no large files are written anywhere.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_key: str = "image",
        label_key: str = "label",
        model_key: str = "generator",
        transform: Optional[Callable] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        # Redirect datasets builder cache to SLURM_TMPDIR; disable Arrow cache.
        tmpdir = os.environ.get("SLURM_TMPDIR") or "/tmp"
        hf_cache = os.path.join(tmpdir, ".hf_datasets_cache")
        os.makedirs(hf_cache, exist_ok=True)
        datasets.config.HF_DATASETS_CACHE = hf_cache
        datasets.disable_caching()

        if data_dir:
            # Load directly from local parquet files — no hub interaction.
            data_dir = os.path.expandvars(data_dir)
            parquet_files = sorted(glob.glob(os.path.join(data_dir, f"{split}-*.parquet")))
            if not parquet_files:
                raise FileNotFoundError(f"No {split}-*.parquet files found in {data_dir!r}")
            self.ds = load_dataset("parquet", data_files={split: parquet_files}, split=split)
        else:
            self.ds = load_dataset(dataset_name, split=split)
        self.image_key = image_key
        self.label_key = label_key
        self.model_key = model_key
        self.transform = transform

    def _label_to_int(self, value) -> int:
        if isinstance(value, str):
            return 1 if value.strip().lower() in {
                "fake",
                "synthetic",
                "full_synthetic",
                "1",
                "tampered",
            } else 0
        return int(value)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        try:
            row = self.ds[idx]
            img = row[self.image_key]
            if not isinstance(img, Image.Image):
                img = Image.open(img)
            img.load()
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self._label_to_int(row[self.label_key])
            model = row.get(self.model_key) or ""
        except Exception:
            img = Image.new("RGB", (224, 224), color=0)
            label = 0
            model = ""
        if self.transform:
            img = self.transform(img)
        return {"image": img, "labels": label, "model_name": str(model)}
