"""
Loss functions for NIDS models.

Consolidates:
  - ``PULoss``: class-form non-negative PU loss with stable gradient handling.
  - ``nnpu_loss``: function-form nnPU using a logistic surrogate over
    pre-split P / U logit batches (used by the 2-way TCN).
  - ``contrastive_nt_xent``: SimCLR-style NT-Xent loss for SSL pretraining.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ─── nnPU (class form, mixed P/U batches) ─────────────


class PULoss(nn.Module):
    """Non-negative Positive-Unlabeled (nnPU) loss for mixed P/U batches."""

    def __init__(self, prior: float, nnpu: bool = True) -> None:
        super().__init__()
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior
        self.nnpu = nnpu
        self._warned_negative = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets.bool()
        unl_mask = ~pos_mask
        n_pos = pos_mask.float().sum().clamp_min(1.0)
        n_unl = unl_mask.float().sum().clamp_min(1.0)

        loss_pos_as_pos = F.softplus(-logits)
        loss_as_neg = F.softplus(logits)

        positive_risk = self.prior * loss_pos_as_pos[pos_mask].sum() / n_pos
        unlabeled_neg_risk = loss_as_neg[unl_mask].sum() / n_unl
        positive_neg_risk = self.prior * loss_as_neg[pos_mask].sum() / n_pos
        negative_risk = unlabeled_neg_risk - positive_neg_risk

        if self.nnpu and negative_risk.item() < 0:
            if not self._warned_negative:
                logger.debug(
                    "nnPU: negative risk clamped (%.4f). Normal during early training.",
                    negative_risk.item(),
                )
                self._warned_negative = True
            return positive_risk - negative_risk.detach() + negative_risk
        return positive_risk + negative_risk


# ─── nnPU (function form, separate P / U logits) ──────


def logistic_loss(logit: torch.Tensor, y_pm1: torch.Tensor) -> torch.Tensor:
    """Logistic surrogate with labels in {+1, -1}: softplus(-y·f)."""
    return F.softplus(-y_pm1 * logit)


def nnpu_loss(
    logit_p: torch.Tensor,
    logit_u: torch.Tensor,
    pi_p: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """nnPU risk estimator over pre-split P and U logit batches."""
    if not 0.0 < pi_p < 1.0:
        raise ValueError(f"pi_p must be in (0, 1), got {pi_p}")

    rp_pos = logistic_loss(logit_p, torch.ones_like(logit_p)).mean()
    rp_neg = logistic_loss(logit_p, -torch.ones_like(logit_p)).mean()
    ru_neg = logistic_loss(logit_u, -torch.ones_like(logit_u)).mean()

    pos_risk = pi_p * rp_pos
    neg_risk = ru_neg - pi_p * rp_neg
    total = pos_risk + torch.clamp(neg_risk, min=0.0)

    stats = {
        "pos_risk": float(pos_risk.detach()),
        "neg_risk_raw": float(neg_risk.detach()),
        "nnpu_risk": float(total.detach()),
    }
    return total, stats


# ─── Contrastive (SimCLR / NT-Xent) ───────────────────


def contrastive_nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """NT-Xent (SimCLR) contrastive loss over a batch of two augmented views."""
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.t()) / temperature
    sim = sim - torch.eye(2 * B, device=z.device) * 1e9

    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)
    return F.cross_entropy(sim, targets)
