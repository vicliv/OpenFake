"""Background-thread DataLoader wrapper that keeps a batch buffer full.

Why this helps
--------------
PyTorch's DataLoader with ``num_workers=N`` and ``prefetch_factor=K`` already
runs augmentation in parallel worker *processes*.  But between steps the main
thread still has to call ``next(iterator)``, which blocks until a worker
delivers a batch.  Under heavy augmentation (JPEG/WebP/AVIF re-encoding) the
workers can momentarily stall the main loop.

``PrefetchLoader`` wraps any iterable DataLoader.  A single daemon thread
continuously pulls batches from the underlying iterator and pushes them into a
``queue.Queue`` of size ``buffer_size``.  The main training loop reads from the
queue, which is almost always non-empty, so ``next()`` never blocks.

Compatibility
-------------
Tensors are kept on CPU; accelerate / DDP handle the H2D transfer as normal.
``__len__`` is forwarded so the Trainer can compute epoch length correctly.
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator


class PrefetchLoader:
    """Wraps a DataLoader and pre-fills a batch buffer using a background thread.

    Args:
        loader:       Any iterable whose ``__iter__`` yields batch dicts.
        buffer_size:  Number of batches to buffer.  Set to 0 to disable
                      (returns the loader unchanged from ``build_trainer``).
    """

    def __init__(self, loader, buffer_size: int = 3) -> None:
        self.loader = loader
        self.buffer_size = buffer_size

    def __getattr__(self, name: str) -> Any:
        """Proxy DataLoader attributes used by Trainer/Accelerate internals."""
        return getattr(self.loader, name)

    def __iter__(self) -> Iterator[Any]:
        buf: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        _DONE = object()

        def _producer() -> None:
            try:
                for batch in self.loader:
                    buf.put(batch)      # blocks if queue is full (back-pressure)
            except BaseException as exc:
                buf.put(exc)
            else:
                buf.put(_DONE)

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = buf.get()
            if item is _DONE:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    def __len__(self) -> int:
        return len(self.loader)
