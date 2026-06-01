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
            "platt": _mod.PlattCalibrator,
            "isotonic": _mod.IsotonicCalibrator,
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


# ─── Platt Scaling ──────────────────────────────────


class PlattCalibrator(BaseCalibrator):
    """
    Learns affine transform: calibrated = sigmoid(a * z + b)
    via L-BFGS on binary cross-entropy over validation logits.
    """

    name = "platt"

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 0.0

    def fit(
        self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200,
    ) -> None:
        logits = logits.detach().cpu().float()
        labels = labels.detach().cpu().float()

        a = torch.tensor(1.0, requires_grad=True)
        b = torch.tensor(0.0, requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [a, b], max_iter=max_iter, line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            cal = torch.sigmoid(a * logits + b)
            loss = F.binary_cross_entropy(cal.clamp(1e-7, 1 - 1e-7), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.a = float(a.item())
        self.b = float(b.item())
        logger.info("Platt scaling: a=%.4f b=%.4f", self.a, self.b)

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * logits + self.b)

    def state_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.a = d["a"]
        self.b = d["b"]


# ─── Isotonic Regression ────────────────────────────


def _pav(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Pool Adjacent Violators — weighted isotonic regression."""
    n = len(y)
    # Each block is [sum_wy, sum_w, start, end]
    blocks: list = []
    for i in range(n):
        blk = [float(w[i] * y[i]), float(w[i]), i, i]
        blocks.append(blk)
        # Merge with previous blocks while violating monotonicity
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blk[0] / blk[1]:
            prev = blocks[-2]
            prev[0] += blk[0]
            prev[1] += blk[1]
            prev[3] = blk[3]
            blocks.pop()
            blk = prev
    result = torch.empty(n)
    for blk in blocks:
        val = blk[0] / blk[1]
        result[blk[2]: blk[3] + 1] = val
    return result


class IsotonicCalibrator(BaseCalibrator):
    """
    Non-parametric isotonic regression calibrator.

    Fits a monotonically non-decreasing step function mapping
    logits → probabilities using the Pool Adjacent Violators algorithm.
    At inference, uses linear interpolation between fitted knots.
    """

    name = "isotonic"

    def __init__(self) -> None:
        self.x_knots: list = []
        self.y_knots: list = []

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.detach().cpu().float()
        labels = labels.detach().cpu().float()

        # Sort by logit value
        order = logits.argsort()
        x_sorted = logits[order]
        y_sorted = labels[order]
        w = torch.ones_like(y_sorted)

        # Run PAV
        y_iso = _pav(y_sorted, w)

        # Deduplicate into knots (store one point per constant block)
        knots_x: list = []
        knots_y: list = []
        i = 0
        n = len(y_iso)
        while i < n:
            val = float(y_iso[i])
            j = i
            while j < n and float(y_iso[j]) == val:
                j += 1
            # Use midpoint of x-range for this block
            knots_x.append(float((x_sorted[i] + x_sorted[j - 1]) / 2))
            knots_y.append(val)
            i = j

        self.x_knots = knots_x
        self.y_knots = knots_y
        logger.info(
            "Isotonic calibration: %d knots, range [%.4f, %.4f]",
            len(knots_x), knots_y[0], knots_y[-1],
        )

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.detach().cpu().float()
        xk = torch.tensor(self.x_knots, dtype=torch.float32)
        yk = torch.tensor(self.y_knots, dtype=torch.float32)

        # Linear interpolation with clamping at boundaries
        flat = x.reshape(-1)
        indices = torch.searchsorted(xk, flat).clamp(1, len(xk) - 1)
        x0 = xk[indices - 1]
        x1 = xk[indices]
        y0 = yk[indices - 1]
        y1 = yk[indices]
        # Avoid division by zero on duplicate knots
        dx = (x1 - x0).clamp(min=1e-8)
        t = ((flat - x0) / dx).clamp(0, 1)
        result = y0 + t * (y1 - y0)
        return result.reshape(logits.shape).clamp(0, 1)

    def state_dict(self) -> Dict[str, Any]:
        return {"x_knots": self.x_knots, "y_knots": self.y_knots}

    def load_state_dict(self, d: Dict[str, Any]) -> None:
        self.x_knots = d["x_knots"]
        self.y_knots = d["y_knots"]


# ─── Auto-selection ─────────────────────────────────


def pick_best_calibrator(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pi_train: float = 0.10,
) -> BaseCalibrator:
    """
    Fit all available calibrators and return the one with lowest log-loss.

    Parameters
    ----------
    logits : raw model logits (val set)
    labels : binary labels
    pi_train : training class prior (for PriorCorrectionCalibrator)

    Returns
    -------
    Best-fitted BaseCalibrator instance.
    """
    candidates: list[BaseCalibrator] = [
        PriorCorrectionCalibrator(pi_train=pi_train),
        PlattCalibrator(),
        IsotonicCalibrator(),
    ]
    best: BaseCalibrator | None = None
    best_ll = float("inf")

    for cal in candidates:
        cal.fit(logits, labels)
        ll = cal.log_loss(logits, labels)
        logger.info("  %s log-loss: %.5f", cal.name, ll)
        if ll < best_ll:
            best_ll = ll
            best = cal

    logger.info("Selected calibrator: %s (log-loss=%.5f)", best.name, best_ll)  # type: ignore[union-attr]
    return best  # type: ignore[return-value]
