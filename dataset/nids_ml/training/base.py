"""
Base trainer abstractions shared by all training loops.

Provides:
  - ``EarlyStopper``: patience-based early stopping helper.
  - ``BaseTrainer``: ABC with shared optimizer, scheduler, gradient, and
    checkpoint management utilities.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)


# ─── Early stopper ───────────────────────────────────


class EarlyStopper:
    """Patience-based early stopping tracker."""

    def __init__(self, patience: int = 3, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: Optional[float] = None
        self.bad = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.bad = 0
            return False
        self.bad += 1
        return self.bad > self.patience


# ─── Base trainer ────────────────────────────────────


class BaseTrainer(ABC):
    """Abstract base for all trainers.

    Subclasses must implement ``train()`` and ``evaluate()``.
    """

    def __init__(self, device: torch.device, training_cfg: Dict[str, Any]) -> None:
        self.device = device
        self.clip_grad = float(training_cfg.get("clip_grad", 1.0))
        self.patience = int(training_cfg.get("patience", 3))

    # ── optimizer / scheduler factories ─────────────

    @staticmethod
    def _init_optimizer(
        params: Iterable[nn.Parameter],
        lr: float,
        weight_decay: float,
        *,
        use_adamw: bool = False,
    ) -> torch.optim.Optimizer:
        cls = torch.optim.AdamW if use_adamw else torch.optim.Adam
        return cls(params, lr=lr, weight_decay=weight_decay)

    @staticmethod
    def _init_scheduler(
        optimizer: torch.optim.Optimizer,
        cfg: Optional[Dict[str, Any]] = None,
        *,
        mode: str = "max",
    ) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
        if cfg is None:
            return None
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 1)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
        )

    # ── gradient step ───────────────────────────────

    def _optim_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        params: Iterable[nn.Parameter],
    ) -> None:
        """Backward pass + clipped gradient step with NaN guard."""
        optimizer.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, self.clip_grad)
            optimizer.step()
        else:
            logger.warning("Non-finite loss detected; skipping optimizer step.")

    # ── scheduler step ──────────────────────────────

    @staticmethod
    def _step_scheduler(
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau],
        metric_value: float,
    ) -> None:
        if scheduler is not None:
            scheduler.step(metric_value)

    # ── checkpoint ──────────────────────────────────

    @staticmethod
    def _save_checkpoint(
        path: Path,
        state_dict: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = dict(state_dict)
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    # ── abstract interface ──────────────────────────

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        ...

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        ...
