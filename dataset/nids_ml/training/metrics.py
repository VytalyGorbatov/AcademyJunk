"""
Metrics for NIDS classifier evaluation.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from ..local_types import Metrics


class MetricUtils:
    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Computes Accuracy, Precision, Recall, and F1 from hard predictions."""
        assert y_true.shape == y_pred.shape

        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))

        eps = 1e-12
        return {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, eps),
            "precision": tp / max(tp + fp, eps),
            "recall": tp / max(tp + fn, eps),
            "f1": 2.0 * tp / max(2.0 * tp + fp + fn, eps),
        }


@torch.no_grad()
def pr_curve_best_f1(
    scores: torch.Tensor,
    y_true: torch.Tensor,
) -> Dict[str, float]:
    """Compute the PR curve and return the threshold that maximises F1."""
    scores = scores.detach().cpu()
    y_true = y_true.detach().cpu()

    if scores.numel() == 0:
        return {
            "best_threshold": 0.5,
            "best_f1": 0.0,
            "pr_auc": 0.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
        }

    idx = torch.argsort(scores, descending=True)
    s = scores[idx]
    y = y_true[idx]

    tp = torch.cumsum(y, dim=0)
    fp = torch.cumsum(1 - y, dim=0)
    fn = tp[-1] - tp

    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

    best_i = int(torch.argmax(f1).item())
    pr_auc = float(torch.trapz(precision, recall).item())

    return {
        "best_threshold": float(s[best_i].item()),
        "best_f1": float(f1[best_i].item()),
        "pr_auc": pr_auc,
        "precision_at_best": float(precision[best_i].item()),
        "recall_at_best": float(recall[best_i].item()),
    }


__all__ = ["MetricUtils", "pr_curve_best_f1"]
