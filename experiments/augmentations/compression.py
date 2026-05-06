"""Social-media compression augmentation.

Simulates the compression pipelines applied by major platforms (Instagram,
Twitter/X, Facebook, YouTube) before serving images. Per call, one compression
type is sampled according to fixed weights:

  JPEG            0.45  – single or double-encode at random quality
  WebP            0.20  – lossy or lossless
  AVIF            0.10  – via pillow-avif-plugin or pillow-heif (falls back to WebP)
  resize + JPEG   0.10  – downsample long edge then re-compress
  no compression  0.15  – passthrough

All encoding is done in-memory; no files are written to disk.
"""

from __future__ import annotations

import io
import random
import warnings
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from config import CompressionConfig

# ---------------------------------------------------------------------------
# Optional AVIF backend detection (done once at import time)
# ---------------------------------------------------------------------------
_AVIF_AVAILABLE = False
_AVIF_SPEED_SUPPORTED = False  # whether the encoder accepts a `speed` kwarg

try:
    import pillow_avif  # noqa: F401
    _AVIF_AVAILABLE = True
except ImportError:
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
        _AVIF_AVAILABLE = True
    except ImportError:
        pass

if _AVIF_AVAILABLE:
    # Probe once whether the installed plugin accepts speed=9
    try:
        _probe = Image.new("RGB", (8, 8))
        _buf = io.BytesIO()
        _probe.save(_buf, format="AVIF", quality=50, speed=9)
        _AVIF_SPEED_SUPPORTED = True
    except Exception:
        pass

_AVIF_WARNED = False


def _ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "P" and "transparency" in img.info:
        return img.convert("RGBA").convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _encode_decode(img: Image.Image, fmt: str, **save_kwargs) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    buf.seek(0)
    out = Image.open(buf)
    out.load()
    return out.convert("RGB")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SocialMediaCompression:
    """Simulate social-media compression for deepfake-detection training.

    Accepts a PIL Image or a uint8 RGB tensor of shape ``(C, H, W)`` or
    ``(H, W, C)``.  Returns in the same format as received.

    Args:
        cfg:  :class:`CompressionConfig` controlling quality ranges and
              sampling probabilities.
        p:    Probability of applying *any* compression.  Set to 0 during
              evaluation to disable the transform entirely.
        seed: Optional integer seed for the internal RNG so augmentation
              is reproducible across runs.
    """

    def __init__(
        self,
        cfg: CompressionConfig,
        p: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.cfg = cfg
        self.p = p
        self._rng = random.Random(seed)

        total = (
            cfg.jpeg_prob + cfg.webp_prob + cfg.avif_prob
            + cfg.resize_jpeg_prob + cfg.no_compression_prob
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Compression sampling probabilities must sum to 1.0, got {total:.4f}"
            )

        self._weights = [
            cfg.jpeg_prob,
            cfg.webp_prob,
            cfg.avif_prob,
            cfg.resize_jpeg_prob,
            cfg.no_compression_prob,
        ]
        self._choices = ["jpeg", "webp", "avif", "resize_jpeg", "none"]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, img: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        if self._rng.random() > self.p:
            return img

        is_tensor = isinstance(img, torch.Tensor)
        chw = False

        if is_tensor:
            arr = img.numpy() if img.dtype == torch.uint8 else (img.float().numpy() * 255).astype("uint8")
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
                chw = True
            pil_img = Image.fromarray(arr.squeeze(-1) if arr.shape[-1] == 1 else arr, "RGB")
        else:
            pil_img = img

        pil_img = _ensure_rgb(pil_img)
        choice = self._rng.choices(self._choices, weights=self._weights, k=1)[0]
        result = self._apply(pil_img, choice)

        if is_tensor:
            result_arr = np.array(result)
            if chw:
                result_arr = result_arr.transpose(2, 0, 1)
            return torch.from_numpy(result_arr.copy())
        return result

    # ------------------------------------------------------------------
    # Compression implementations
    # ------------------------------------------------------------------

    def _apply(self, img: Image.Image, choice: str) -> Image.Image:
        try:
            return self._apply_unsafe(img, choice)
        except Exception:
            # On any encoding failure fall back to a safe JPEG round-trip
            q = self._rng.randint(self.cfg.jpeg_quality[0], self.cfg.jpeg_quality[1])
            try:
                return _encode_decode(img, "JPEG", quality=q)
            except Exception:
                return img  # last resort: return original

    def _apply_unsafe(self, img: Image.Image, choice: str) -> Image.Image:
        cfg = self.cfg
        rng = self._rng

        if choice == "jpeg":
            q = rng.randint(cfg.jpeg_quality[0], cfg.jpeg_quality[1])
            out = _encode_decode(img, "JPEG", quality=q)
            if rng.random() < cfg.jpeg_double_encode_p:
                q2 = rng.randint(cfg.jpeg_double_quality[0], cfg.jpeg_double_quality[1])
                out = _encode_decode(out, "JPEG", quality=q2)
            return out

        if choice == "webp":
            # method=0: fastest WebP encoder (~3-5× speedup vs default method=4)
            if rng.random() < cfg.webp_lossless_p:
                return _encode_decode(img, "WEBP", lossless=True, method=0)
            q = rng.randint(cfg.webp_quality[0], cfg.webp_quality[1])
            return _encode_decode(img, "WEBP", quality=q, method=0)

        if choice == "avif":
            if _AVIF_AVAILABLE:
                q = rng.randint(cfg.avif_quality[0], cfg.avif_quality[1])
                # speed=9: fastest AVIF encoder (100-500ms → ~20-50ms per image)
                avif_kwargs = {"quality": q, **({"speed": 9} if _AVIF_SPEED_SUPPORTED else {})}
                try:
                    return _encode_decode(img, "AVIF", **avif_kwargs)
                except Exception:
                    pass
            # Graceful fallback to WebP
            global _AVIF_WARNED
            if not _AVIF_WARNED:
                warnings.warn(
                    "AVIF encoding unavailable (install pillow-avif-plugin or pillow-heif); "
                    "falling back to WebP for AVIF samples.",
                    stacklevel=2,
                )
                _AVIF_WARNED = True
            q = rng.randint(cfg.webp_quality[0], cfg.webp_quality[1])
            return _encode_decode(img, "WEBP", quality=q)

        if choice == "resize_jpeg":
            w, h = img.size
            long_edge = max(w, h)
            target = rng.randint(cfg.resize_jpeg_long_edge[0], cfg.resize_jpeg_long_edge[1])
            if long_edge > target:
                scale = target / long_edge
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                img = img.resize((new_w, new_h), Image.BICUBIC)
            q = rng.randint(cfg.resize_jpeg_quality[0], cfg.resize_jpeg_quality[1])
            return _encode_decode(img, "JPEG", quality=q)

        # "none" — passthrough
        return img
