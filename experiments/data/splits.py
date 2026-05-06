"""Utilities to load metadata CSVs, resolve absolute paths, and build datasets."""

from __future__ import annotations

import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from pathlib import Path

from config import Config
from data.dataset import OpenFakeDataset
from data.hf_dataset import HFMapDataset
from data.streaming_dataset import StreamingHFDataset

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Per-split metadata schema
# ---------------------------------------------------------------------------

# path_prefix:
#   ""      → join directly with openfake_root  (train/test)
#   "civit" → join with openfake_root/civit/    (civit)
#   None    → treat path as absolute / use as-is after fallback (reddit)

_SPLIT_INFO: dict = {
    "train":  {"path_col": "file_name",   "label_col": "label",  "path_prefix": ""},
    "test":   {"path_col": "file_name",   "label_col": "label",  "path_prefix": ""},
    "reddit": {"path_col": "image_path",  "label_col": "label",  "path_prefix": None},
    "civit":  {"path_col": "file_name",   "label_col": None,     "path_prefix": "civit",
               "default_label": 1},
}


def _csv_path(split: str, root: str) -> str:
    if split.endswith(".csv") or os.path.isabs(split):
        if os.path.isabs(split):
            return split
        cwd_path = Path(split).resolve()
        if cwd_path.exists():
            return str(cwd_path)
        return str((REPO_ROOT / split).resolve())
    return os.path.join(root, split, "metadata.csv")


def _resolve_path(raw: str, root: str, prefix: Optional[str]) -> str:
    """Return an absolute filesystem path for a metadata row."""
    if os.path.isabs(raw):
        return raw
    if prefix is None:
        # reddit: paths may be stored as absolute or relative to root
        candidate = os.path.join(root, raw)
        return candidate
    if prefix == "":
        return os.path.join(root, raw)
    return os.path.join(root, prefix, raw)


def _infer_split_info(split: str, csv: str) -> dict:
    info = _SPLIT_INFO.get(split)
    if info is not None:
        return info
    return {"path_col": "file_name", "label_col": "label", "path_prefix": None}


def load_split_samples(split: str, root: str) -> List[Tuple[str, int, str]]:
    """Read a metadata CSV and return ``[(abs_path, label, model), ...]``."""
    csv = _csv_path(split, root)
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Metadata CSV not found: {csv}")
    info = _infer_split_info(split, csv)

    path_col: str = info["path_col"]
    label_col: Optional[str] = info["label_col"]
    prefix: Optional[str] = info["path_prefix"]
    default_label: int = info.get("default_label", 0)
    dtype = {path_col: "string"}
    if label_col is not None:
        dtype[label_col] = "string"
    df = pd.read_csv(csv, dtype=dtype, low_memory=False)

    paths = df[path_col].astype(str).apply(
        lambda p: _resolve_path(p, root, prefix)
    ).tolist()

    exists = [os.path.exists(path) for path in paths]
    missing = len(paths) - sum(exists)
    if missing:
        print(f"[WARN] {csv}: skipping {missing:,}/{len(paths):,} rows with missing image files")
        df = df.loc[exists].reset_index(drop=True)
        paths = [path for path, ok in zip(paths, exists) if ok]

    if label_col and label_col in df.columns:
        labels = (df[label_col].str.strip().str.lower() == "fake").astype(int).tolist()
    else:
        labels = [default_label] * len(df)
    if "model" in df.columns:
        models = df["model"].fillna("").astype(str).tolist()
    elif split == "reddit":
        models = ["reddit"] * len(df)
    else:
        models = [""] * len(df)

    return list(zip(paths, labels, models))


def _stratified_sample(
    samples: List[Tuple[str, int, str]],
    max_samples: int,
    seed: int = 42,
) -> List[Tuple[str, int, str]]:
    """Return a deterministic label-balanced sample, preserving all classes."""
    if len(samples) <= max_samples:
        return samples

    by_label: Dict[int, List[Tuple[str, int, str]]] = {}
    for sample in samples:
        by_label.setdefault(sample[1], []).append(sample)

    if len(by_label) < 2:
        rng = random.Random(seed)
        sampled = list(samples)
        rng.shuffle(sampled)
        return sampled[:max_samples]

    rng = random.Random(seed)
    for label_samples in by_label.values():
        rng.shuffle(label_samples)

    labels = sorted(by_label)
    base_quota = max_samples // len(labels)
    remainder = max_samples % len(labels)
    selected: List[Tuple[str, int, str]] = []
    leftovers: List[Tuple[str, int, str]] = []

    for i, label in enumerate(labels):
        quota = base_quota + (1 if i < remainder else 0)
        label_samples = by_label[label]
        selected.extend(label_samples[:quota])
        leftovers.extend(label_samples[quota:])

    if len(selected) < max_samples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected


def _build_external_val_samples(
    config: Config,
    root: str,
    max_samples: int = 5000,
) -> List[Tuple[str, int, str]]:
    val_samples: List[Tuple[str, int, str]] = []
    for split in config.data.test_splits[:1]:
        val_samples.extend(_stratified_sample(load_split_samples(split, root), max_samples))
    return val_samples


def build_datasets(
    config: Config,
    train_transform: Optional[Callable],
    eval_transform: Optional[Callable],
) -> Tuple[OpenFakeDataset, OpenFakeDataset, Dict[str, OpenFakeDataset]]:
    """Build train, val, and test datasets from the config.

    The processor (resize + normalize) is NOT applied here; use
    :class:`data.collator.OpenFakeCollator` as the Trainer's
    ``data_collator`` to apply it at batch-collation time.

    Returns:
        train_ds:    95% of combined training splits (with ``train_transform``).
        val_ds:      Remaining 5% (with ``eval_transform``).
        test_ds_map: ``{split_name: OpenFakeDataset}`` for each test split.
    """
    root = config.data.openfake_root

    # ---- Collect and shuffle all training samples ----
    if config.data.hf_train:
        ht = config.data.hf_train
        train_ds = HFMapDataset(
            dataset_name=ht.get("dataset_name", ""),
            split=ht.get("split", "train"),
            image_key=ht.get("image_key", "image"),
            label_key=ht.get("label_key", "label"),
            model_key=ht.get("model_key", "generator"),
            transform=train_transform,
            data_dir=ht.get("data_dir"),
        )
        val_samples = _build_external_val_samples(config, root)
        val_ds = OpenFakeDataset(val_samples, transform=eval_transform)
    elif config.data.streaming_train:
        st = config.data.streaming_train
        train_ds = StreamingHFDataset(
            dataset_name=st.get("dataset_name", ""),
            split=st.get("split", "train"),
            image_key=st.get("image_key", "image"),
            label_key=st.get("label_key", "label"),
            model_key=st.get("model_key", "generator"),
            shuffle_buffer=st.get("shuffle_buffer", 10000),
            transform=train_transform,
            data_dir=st.get("data_dir"),
        )
        val_samples = _build_external_val_samples(config, root)
        val_ds = OpenFakeDataset(val_samples, transform=eval_transform)
    else:
        all_samples: List[Tuple[str, int]] = []
        for split in config.data.train_splits:
            all_samples.extend(load_split_samples(split, root))

        if config.data.val_split:
            val_samples = load_split_samples(config.data.val_split, root)
            train_samples = all_samples
        else:
            rng = random.Random(42)
            rng.shuffle(all_samples)
            n_val = max(1, int(len(all_samples) * config.data.val_fraction))
            val_samples = all_samples[:n_val]
            train_samples = all_samples[n_val:]

        train_ds = OpenFakeDataset(train_samples, transform=train_transform)
        val_ds = OpenFakeDataset(val_samples, transform=eval_transform)

    # ---- Test splits ----
    test_ds_map: Dict[str, OpenFakeDataset] = {}
    test_splits = config.data.test_sets or config.data.test_splits
    for split in test_splits:
        test_samples = load_split_samples(split, root)
        name = os.path.splitext(os.path.basename(split))[0] if split.endswith(".csv") else split
        test_ds_map[name] = OpenFakeDataset(test_samples, transform=eval_transform)

    try:
        train_size = f"{len(train_ds):,}"
    except TypeError:
        train_size = "streaming"
    print(f"Dataset sizes — train: {train_size}  val: {len(val_ds):,}")
    for name, ds in test_ds_map.items():
        print(f"  test/{name}: {len(ds):,}")

    return train_ds, val_ds, test_ds_map
