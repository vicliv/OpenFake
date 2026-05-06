from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(pred) -> dict:
    """Compute accuracy, precision, recall, F1, and AUC-ROC from Trainer predictions."""
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])

    probs = torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()[:, 1]
    labels = np.asarray(pred.label_ids)

    finite_mask = np.isfinite(probs)
    if not np.all(finite_mask):
        n_dropped = np.size(probs) - np.count_nonzero(finite_mask)
        print(f"[WARN] Dropping {n_dropped} non-finite probability entries from metrics")
        probs = probs[finite_mask]
        labels = labels[finite_mask]

    if probs.size == 0:
        return dict(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, auc_roc=float("nan"))

    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    auc_roc = (
        roc_auc_score(labels, probs)
        if len(np.unique(labels)) > 1
        else float("nan")
    )

    return dict(accuracy=acc, precision=precision, recall=recall, f1=f1, auc_roc=auc_roc)


def _positive_probs(predictions) -> np.ndarray:
    logits = predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits)
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.softmax(torch.from_numpy(logits).float(), dim=-1).numpy()[:, 1]


def compute_per_model_metrics(predictions, labels, model_names) -> pd.DataFrame:
    """Return per-model binary metrics for fake-vs-real predictions."""
    probs = _positive_probs(predictions)
    labels = np.asarray(labels).astype(int)
    model_names = np.asarray(model_names).astype(str)
    finite = np.isfinite(probs)
    probs, labels, model_names = probs[finite], labels[finite], model_names[finite]
    preds = (probs >= 0.5).astype(int)

    rows = []
    for model in sorted(pd.unique(model_names)):
        mask = model_names == model
        y_true = labels[mask]
        y_pred = preds[mask]
        y_prob = probs[mask]
        pos = y_true == 1
        neg = y_true == 0
        rows.append(
            {
                "model": model,
                "n": int(mask.sum()),
                "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
                "tpr": float(((y_pred[pos] == 1).sum() / pos.sum()) if pos.sum() else np.nan),
                "tnr": float(((y_pred[neg] == 0).sum() / neg.sum()) if neg.sum() else np.nan),
                "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0,
                "roc_auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)
