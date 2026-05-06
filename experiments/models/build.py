from __future__ import annotations

import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import ModelConfig

MODEL_REGISTRY: dict[str, str] = {
    "resnet-152":      "microsoft/resnet-152",
    "swinv2-small":    "microsoft/swinv2-small-patch4-window16-256",
    "vit-base":        "google/vit-base-patch16-224",
    "convnextv2-base": "facebook/convnextv2-base-22k-224",
}


def build_model(cfg: ModelConfig):
    """Return ``(processor, model)`` ready for 2-class binary classification.

    If ``cfg.pretrained`` is False the model weights are re-initialised from
    scratch after the classifier head is replaced.
    """
    hf_name = MODEL_REGISTRY.get(cfg.name)
    if hf_name is None:
        raise ValueError(
            f"Unknown model '{cfg.name}'. Available: {list(MODEL_REGISTRY)}"
        )

    processor = AutoImageProcessor.from_pretrained(
        hf_name, cache_dir=cfg.cache_dir, backend="torchvision"
    )
    model = AutoModelForImageClassification.from_pretrained(
        hf_name, cache_dir=cfg.cache_dir
    )

    # Replace classifier head for 2-class binary output
    model.num_labels = cfg.num_labels
    model.config.num_labels = cfg.num_labels
    model.config.id2label = {0: "real", 1: "fake"}
    model.config.label2id = {"real": 0, "fake": 1}

    name = cfg.name
    if name == "resnet-152":
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(model.config.hidden_sizes[-1], cfg.num_labels),
        )
    elif name == "swinv2-small":
        model.classifier = nn.Linear(model.swinv2.num_features, cfg.num_labels)
    elif name == "vit-base":
        model.classifier = nn.Linear(model.config.hidden_size, cfg.num_labels)
    elif name == "convnextv2-base":
        model.classifier = nn.Linear(model.config.hidden_sizes[-1], cfg.num_labels)

    if not cfg.pretrained:
        model.init_weights()

    return processor, model
