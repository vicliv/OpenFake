from __future__ import annotations

import os
import dataclasses
import typing
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    jpeg_quality: List[int] = field(default_factory=lambda: [55, 95])
    jpeg_double_encode_p: float = 0.3
    jpeg_double_quality: List[int] = field(default_factory=lambda: [50, 85])
    webp_quality: List[int] = field(default_factory=lambda: [65, 90])
    webp_lossless_p: float = 0.05
    avif_quality: List[int] = field(default_factory=lambda: [40, 80])
    resize_jpeg_long_edge: List[int] = field(default_factory=lambda: [720, 1080])
    resize_jpeg_quality: List[int] = field(default_factory=lambda: [65, 85])
    # Sampling weights — must sum to 1.0
    jpeg_prob: float = 0.45
    webp_prob: float = 0.20
    avif_prob: float = 0.10
    resize_jpeg_prob: float = 0.10
    no_compression_prob: float = 0.15


@dataclass
class AugmentConfig:
    use_compression: bool = True
    compression_p: float = 1.0
    compression_seed: Optional[int] = None
    compression: CompressionConfig = field(default_factory=CompressionConfig)

    random_resized_crop: bool = True
    random_resized_crop_size: int = 256
    random_resized_crop_scale: List[float] = field(default_factory=lambda: [0.5, 1.0])
    random_resized_crop_ratio: List[float] = field(default_factory=lambda: [0.5, 2.0])

    color_jitter: bool = True
    color_jitter_p: float = 0.8
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.5
    color_jitter_saturation: float = 0.5

    random_rotation: bool = True
    random_rotation_p: float = 0.5
    random_rotation_degrees: float = 15.0

    random_horizontal_flip: bool = True
    random_horizontal_flip_p: float = 0.1

    gaussian_blur: bool = True
    gaussian_blur_p: float = 0.5
    gaussian_blur_kernel: List[int] = field(default_factory=lambda: [5, 5])
    gaussian_blur_sigma: List[float] = field(default_factory=lambda: [0.1, 0.5])


@dataclass
class DataConfig:
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    test_splits: List[str] = field(default_factory=lambda: ["test"])
    test_sets: List[str] = field(default_factory=list)
    val_fraction: float = 0.05
    val_split: Optional[str] = None       # CSV path for fixed val set; overrides val_fraction
    openfake_root: Optional[str] = None   # required for directory-based train/test metadata
    num_workers: int = 8
    streaming_train: Optional[dict] = None
    hf_train: Optional[dict] = None


@dataclass
class ModelConfig:
    name: str = "swinv2-small"
    pretrained: bool = True
    cache_dir: Optional[str] = None       # defaults to $SCRATCH/.cache at runtime
    num_labels: int = 2


@dataclass
class TrainingConfig:
    output_dir: Optional[str] = None      # set in YAML/CLI, or auto-derived under ./runs
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 8e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    eval_steps: int = 2000
    save_steps: int = 2000
    save_total_limit: int = 3
    fp16: bool = True
    bf16: bool = False
    dataloader_num_workers: int = 16
    dataloader_prefetch_factor: int = 4
    dataloader_persistent_workers: bool = True
    prefetch_buffer_size: int = 3   # batches buffered by PrefetchLoader (0 = disabled)
    report_to: str = "wandb"
    wandb_project: str = "openfake"
    wandb_run_name: Optional[str] = None
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_auc_roc"
    greater_is_better: bool = True
    logging_steps: int = 100
    ddp_find_unused_parameters: bool = False
    max_steps: int = -1


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _from_dict(cls, data: dict):
    """Recursively construct a dataclass from a plain dict."""
    if data is None:
        data = {}
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        hint = hints[f.name]

        # Unwrap Optional[X] → X
        origin = typing.get_origin(hint)
        if origin is typing.Union:
            args = [a for a in typing.get_args(hint) if a is not type(None)]
            if len(args) == 1:
                hint = args[0]

        if dataclasses.is_dataclass(hint) and isinstance(val, dict):
            val = _from_dict(hint, val)

        kwargs[f.name] = val
    return cls(**kwargs)


def load_config(*paths: str) -> Config:
    """Load and merge one or more YAML config files into a Config dataclass.

    Later files override earlier ones (deep merge on nested dicts).
    """
    merged: dict = {}
    for path in paths:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        merged = _deep_merge(merged, data)
    cfg = _from_dict(Config, merged)

    # Apply environment-variable defaults for paths not set in YAML. These are
    # portable fallbacks; publication configs should pass explicit paths.
    scratch = os.environ.get("SCRATCH")
    if cfg.data.openfake_root is None:
        cfg.data.openfake_root = os.path.join(scratch, "OpenFake") if scratch else "OpenFake"
    if cfg.model.cache_dir is None:
        cfg.model.cache_dir = os.path.join(scratch, ".cache") if scratch else ".cache"

    return cfg
