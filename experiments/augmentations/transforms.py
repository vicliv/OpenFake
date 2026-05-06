from __future__ import annotations

from torchvision import transforms

from config import AugmentConfig
from augmentations.compression import SocialMediaCompression


def build_train_transform(cfg: AugmentConfig) -> transforms.Compose:
    """Build the training augmentation pipeline.

    Order: spatial augmentations first, compression last.  Compressing the
    already-cropped image (a) keeps dimensions small so VP8/WebP encoding
    never overflows, and (b) preserves compression artifacts on the exact
    pixels the model will see rather than washing them out via resampling.
    """
    ops = []

    if cfg.random_resized_crop:
        ops.append(
            transforms.RandomResizedCrop(
                size=cfg.random_resized_crop_size,
                scale=tuple(cfg.random_resized_crop_scale),
                ratio=tuple(cfg.random_resized_crop_ratio),
            )
        )

    if cfg.color_jitter:
        ops.append(
            transforms.RandomApply(
                [transforms.ColorJitter(
                    brightness=cfg.color_jitter_brightness,
                    contrast=cfg.color_jitter_contrast,
                    saturation=cfg.color_jitter_saturation,
                )],
                p=cfg.color_jitter_p,
            )
        )

    if cfg.random_rotation:
        ops.append(
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=cfg.random_rotation_degrees)],
                p=cfg.random_rotation_p,
            )
        )

    if cfg.random_horizontal_flip:
        ops.append(transforms.RandomHorizontalFlip(p=cfg.random_horizontal_flip_p))

    if cfg.gaussian_blur:
        ops.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(
                    kernel_size=tuple(cfg.gaussian_blur_kernel),
                    sigma=tuple(cfg.gaussian_blur_sigma),
                )],
                p=cfg.gaussian_blur_p,
            )
        )

    # Compression last — operates on the already-cropped image
    if cfg.use_compression:
        ops.append(
            SocialMediaCompression(
                cfg=cfg.compression,
                p=cfg.compression_p,
                seed=cfg.compression_seed,
            )
        )

    return transforms.Compose(ops)


def build_eval_transform(cfg: AugmentConfig) -> transforms.Compose:
    """Eval transform: no augmentation. Processor handles resize/normalize."""
    return transforms.Compose([])
