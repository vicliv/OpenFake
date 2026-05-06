"""Collator that applies the image processor to a batch of PIL images.

Running the processor (resize + normalize → float tensor) in the main-process
collator instead of inside each worker has two benefits:

  1. Workers only transmit uint8 PIL images through shared memory (~200 KB each
     vs ~800 KB for a float32 tensor), reducing IPC pressure.
  2. Workers spend 100% of their time on the expensive part: disk I/O and
     compression augmentation.
"""

from __future__ import annotations

import torch


class OpenFakeCollator:
    """Batch PIL images through a HuggingFace processor and stack labels.

    Args:
        processor:  Any HuggingFace ``ImageProcessor`` (AutoImageProcessor,
                    ViTImageProcessor, …).  Called with ``images=list_of_pils``.
    """

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, examples: list) -> dict:
        images = [e["image"] for e in examples]
        labels = torch.tensor([e["labels"] for e in examples], dtype=torch.long)
        out = self.processor(images=images, return_tensors="pt")
        out["labels"] = labels
        return out
