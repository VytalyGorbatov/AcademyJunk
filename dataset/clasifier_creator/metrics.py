import numpy as np

from .local_types import Metrics


class MetricUtils:
    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Computes Accuracy, Precision, Recall, and F1-Score."""
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
