from __future__ import annotations

import glob
import os
import random
from typing import Callable, Optional

import torch
from PIL import Image
from datasets import load_dataset


class StreamingHFDataset(torch.utils.data.IterableDataset):
    """HF streaming wrapper returning {"image": PIL, "labels": int} items.

    When ``data_dir`` is provided the dataset is streamed from local parquet
    files (glob ``data_dir/train-*.parquet``) — no network access required.
    Otherwise it streams from the HuggingFace hub via ``dataset_name``.
    """

    def __init__(
        self,
        dataset_name: str = "",
        split: str = "train",
        image_key: str = "image",
        label_key: str = "label",
        model_key: str = "generator",
        shuffle_buffer: int = 10000,
        seed: int = 42,
        transform: Optional[Callable] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.image_key = image_key
        self.label_key = label_key
        self.model_key = model_key
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.transform = transform
        self.data_dir = data_dir

    def _get_value(self, row: dict, key: str, fallback_keys: tuple[str, ...]):
        if key in row:
            return row[key]
        for fallback_key in fallback_keys:
            if fallback_key in row:
                return row[fallback_key]
        available = ", ".join(row.keys())
        raise KeyError(f"Missing key '{key}'. Available keys: {available}")

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

    def __iter__(self):
        if self.data_dir:
            parquet_files = sorted(glob.glob(os.path.join(self.data_dir, "train-*.parquet")))
            if not parquet_files:
                raise FileNotFoundError(
                    f"No parquet files matching train-*.parquet found in {self.data_dir!r}. "
                    "Run slurm/download_sofake.sh first."
                )
            ds = load_dataset("parquet", data_files={"train": parquet_files}, split="train", streaming=True)
        else:
            ds = load_dataset(self.dataset_name, split=self.split, streaming=True)
        if self.shuffle_buffer:
            worker = torch.utils.data.get_worker_info()
            seed = self.seed + (worker.id if worker else 0)
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=seed)
        skipped = 0
        for row in ds:
            try:
                img = self._get_value(row, self.image_key, ("img", "jpg", "png"))
                if not isinstance(img, Image.Image):
                    img = Image.open(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                label_value = self._get_value(row, self.label_key, ("labels", "class", "target", "is_fake"))
                label = self._label_to_int(label_value)
                model = row.get(self.model_key) or row.get("source") or row.get("model") or ""
                yield {"image": img, "labels": label, "model_name": str(model)}
            except Exception as exc:
                skipped += 1
                if skipped <= 10:
                    print(f"[WARN] Skipping streaming sample: {exc}")
                if skipped >= 100:
                    raise RuntimeError(
                        "Skipped 100 consecutive streaming samples. "
                        "Check streaming_train image_key/label_key/model_key against the dataset schema."
                    ) from exc
                continue
            else:
                skipped = 0
