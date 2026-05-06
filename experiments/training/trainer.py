from __future__ import annotations

import inspect
import os
from typing import Callable, Optional

import torch
from transformers import Trainer, TrainingArguments, default_data_collator
from transformers.trainer_utils import get_last_checkpoint

from config import Config
from training.prefetch import PrefetchLoader


# ---------------------------------------------------------------------------
# Trainer subclass
# ---------------------------------------------------------------------------

class OpenFakeTrainer(Trainer):
    """Trainer that wraps the train DataLoader with a background-thread buffer.

    The ``PrefetchLoader`` keeps ``prefetch_buffer_size`` batches ready so
    the GPU never idles waiting for the next augmented batch.
    """

    def __init__(
        self,
        *args,
        prefetch_buffer_size: int = 3,
        eval_dataloader_num_workers: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prefetch_buffer_size = prefetch_buffer_size
        self._eval_dataloader_num_workers = eval_dataloader_num_workers

    def get_train_dataloader(self):
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            old_num_workers = self.args.dataloader_num_workers
            old_persistent_workers = getattr(self.args, "dataloader_persistent_workers", None)
            old_prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
            self.args.dataloader_num_workers = 0
            if old_persistent_workers is not None:
                self.args.dataloader_persistent_workers = False
            if hasattr(self.args, "dataloader_prefetch_factor"):
                self.args.dataloader_prefetch_factor = None
            try:
                loader = super().get_train_dataloader()
            finally:
                self.args.dataloader_num_workers = old_num_workers
                if old_persistent_workers is not None:
                    self.args.dataloader_persistent_workers = old_persistent_workers
                if hasattr(self.args, "dataloader_prefetch_factor"):
                    self.args.dataloader_prefetch_factor = old_prefetch_factor
            return loader

        loader = super().get_train_dataloader()
        if self._prefetch_buffer_size > 0:
            return PrefetchLoader(loader, buffer_size=self._prefetch_buffer_size)
        return loader

    def _build_loader_without_worker_fanout(self, factory, *args, **kwargs):
        old_num_workers = self.args.dataloader_num_workers
        old_persistent_workers = getattr(self.args, "dataloader_persistent_workers", None)
        old_prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
        self.args.dataloader_num_workers = self._eval_dataloader_num_workers
        if old_persistent_workers is not None:
            self.args.dataloader_persistent_workers = False
        if hasattr(self.args, "dataloader_prefetch_factor"):
            self.args.dataloader_prefetch_factor = None
        try:
            return factory(*args, **kwargs)
        finally:
            self.args.dataloader_num_workers = old_num_workers
            if old_persistent_workers is not None:
                self.args.dataloader_persistent_workers = old_persistent_workers
            if hasattr(self.args, "dataloader_prefetch_factor"):
                self.args.dataloader_prefetch_factor = old_prefetch_factor

    def get_eval_dataloader(self, eval_dataset=None):
        return self._build_loader_without_worker_fanout(super().get_eval_dataloader, eval_dataset)

    def get_test_dataloader(self, test_dataset):
        return self._build_loader_without_worker_fanout(super().get_test_dataloader, test_dataset)


# ---------------------------------------------------------------------------
# TrainingArguments builder
# ---------------------------------------------------------------------------

def _supports_arg(arg_name: str) -> bool:
    return arg_name in inspect.signature(TrainingArguments.__init__).parameters


def _compute_warmup_steps(cfg: Config, train_ds_len: int) -> int:
    if cfg.training.max_steps and cfg.training.max_steps > 0:
        return max(1, int(cfg.training.max_steps * cfg.training.warmup_ratio))
    n_devices = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = max(1, train_ds_len // (cfg.training.per_device_train_batch_size * n_devices))
    total_steps = steps_per_epoch * cfg.training.num_train_epochs
    return max(1, int(total_steps * cfg.training.warmup_ratio))


def build_training_args(cfg: Config, warmup_steps: int = 0) -> TrainingArguments:
    t = cfg.training
    kwargs: dict = dict(
        output_dir=t.output_dir,
        num_train_epochs=t.num_train_epochs,
        per_device_train_batch_size=t.per_device_train_batch_size,
        per_device_eval_batch_size=t.per_device_eval_batch_size,
        learning_rate=t.learning_rate,
        weight_decay=t.weight_decay,
        warmup_steps=warmup_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="steps",
        save_steps=t.save_steps,
        save_total_limit=t.save_total_limit,
        fp16=t.fp16,
        bf16=t.bf16,
        dataloader_num_workers=t.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,   # keep "image" key so OpenFakeCollator sees it
        report_to=t.report_to,
        run_name=t.wandb_run_name,
        load_best_model_at_end=t.load_best_model_at_end,
        metric_for_best_model=t.metric_for_best_model,
        greater_is_better=t.greater_is_better,
        logging_steps=t.logging_steps,
        ddp_find_unused_parameters=t.ddp_find_unused_parameters,
        max_steps=t.max_steps,
    )
    # Added in transformers ≥4.39; skip gracefully on older installs
    if _supports_arg("dataloader_prefetch_factor"):
        kwargs["dataloader_prefetch_factor"] = t.dataloader_prefetch_factor
    if _supports_arg("dataloader_persistent_workers"):
        kwargs["dataloader_persistent_workers"] = t.dataloader_persistent_workers
    return TrainingArguments(**kwargs)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_trainer(
    cfg: Config,
    model,
    train_ds,
    val_ds,
    compute_metrics: Callable,
    data_collator=None,
) -> tuple[OpenFakeTrainer, Optional[str]]:
    """Build an ``OpenFakeTrainer`` and detect the latest checkpoint for auto-resume.

    Args:
        data_collator:  Collator that converts worker outputs to model inputs.
                        Should be an :class:`data.collator.OpenFakeCollator`.
                        Falls back to ``default_data_collator`` if not provided.

    Returns:
        trainer:      Ready-to-use trainer with prefetch buffer.
        resume_ckpt:  Path to the latest checkpoint, or ``None``.
    """
    try:
        train_len = len(train_ds)
    except TypeError:
        train_len = cfg.training.max_steps * cfg.training.per_device_train_batch_size
    warmup_steps = _compute_warmup_steps(cfg, train_len)
    print(f"Warmup steps: {warmup_steps}")
    training_args = build_training_args(cfg, warmup_steps=warmup_steps)

    trainer = OpenFakeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator if data_collator is not None else default_data_collator,
        compute_metrics=compute_metrics,
        prefetch_buffer_size=cfg.training.prefetch_buffer_size,
        eval_dataloader_num_workers=4,
    )

    resume_ckpt: Optional[str] = None
    output_dir = cfg.training.output_dir
    if output_dir and os.path.isdir(output_dir):
        resume_ckpt = get_last_checkpoint(output_dir)
        if resume_ckpt:
            print(f"Resuming from checkpoint: {resume_ckpt}")

    return trainer, resume_ckpt
