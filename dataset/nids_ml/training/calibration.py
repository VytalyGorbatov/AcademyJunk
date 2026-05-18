"""
Post-training risk-score calibration.

Transforms raw model logits into calibrated probabilities so that
a threshold near 0.5 separates attack from benign.
"""
from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _logit(p: float) -> float:
    """Log-odds: ln(p / (1-p))."""
    p = max(min(p, 1 - 1e-7), 1e-7)
    return math.log(p / (1.0 - p))


def _log_loss(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Binary cross-entropy (lower is better)."""
    probs = probs.clamp(1e-7, 1 - 1e-7)
    return float(F.binary_cross_entropy(probs, labels.float()).item())


# ─── Base ────────────────────────────────────────────


class BaseCalibrator(ABC):
    """Interface for logit calibrators."""

    name: str = "base"

    @abstractmethod
    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None: ...

    @abstractmethod
    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities in [0, 1]."""
        ...

    @abstractmethod
    def state_dict(self) -> Dict[str, Any]: ...

    @abstractmethod
    def load_state_dict(self, d: Dict[str, Any]) -> None: ...

    def save(self, path: Path) -> None:
        data = {"type": self.name, **self.state_dict()}
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: Path) -> "BaseCalibrator":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        from . import calibration as _mod
        cls_map = {
            "prior_correction": _mod.PriorCorrectionCalibrator,
        }
        ctype = data["type"]
        if ctype not in cls_map:
            raise ValueError(f"Unknown calibrator type: {ctype}")
        cal = cls_map[ctype].__new__(cls_map[ctype])
        cal.name = ctype
        cal.load_state_dict(data)
        return cal

    def log_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute log-loss of calibrated predictions."""
        return _log_loss(self.transform(logits), labels)


# ─── Prior Correction ───────────────────────────────


class PriorCorrectionCalibrator(BaseCalibrator):
    """
    Logit-shift calibration based on train vs target class prior.

    z_corrected = z_raw + logit(pi_target) - logit(pi_train)
    s_corrected = sigmoid(z_corrected)
    """

    name = "prior_correction"

    def __init__(
        self,
        pi_train: float = 0.10,
        pi_target: Optional[float] = None,
    ) -> None:
        self.pi_train = pi_train
        self.pi_target = pi_target
        self.shift = 0.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if self.pi_target is None:
            self.pi_target = float(labels.float().mean().item())
        self.shift = _logit(self.pi_target) - _logit(self.pi_train)
        logger.info(
            "PriorCorrection: pi_train=%.4f pi_target=%.4f shift=%.4f",
            self.pi_train, self.pi_target, self.shift,
        )

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits + self.shift)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "pi_train": self.pi_train,
            "pi_target": self.pi_target,
            "shift": self.shift,
        }

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.pi_train = d["pi_train"]
        self.pi_target = d["pi_target"]
        self.shift = d["shift"]
