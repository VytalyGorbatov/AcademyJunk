"""
Non-negative PU (nnPU) loss for Positive-Unlabeled learning.

Reference:
    Kiryo et al., "Positive-Unlabeled Learning with Non-Negative Risk Estimator", NeurIPS 2017.

Terminology
-----------
- **Positive (P):** labelled as 1 (e.g. `alerted = 1` — IDS confirmed).
- **Unlabeled (U):** labelled as 0, but actually a *mixture* of true negatives and
  hidden positives (attacks the IDS missed).
- **Class prior π:** estimated proportion of true positives in the *entire*
  population (P + U).  Must be supplied or estimated externally.

Standard binary cross-entropy incorrectly treats U as truly negative.
nnPU corrects for the hidden positives in U:

    L_pu = π · E_P[ℓ(f(x), 1)]  +  max(0,  E_U[ℓ(f(x), 0)] − π · E_P[ℓ(f(x), 0)])

The `max(0, …)` clamp is the "non-negative" safeguard that prevents the
negative-risk overfitting failure mode of standard uPU.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class PULoss(nn.Module):
    """Non-negative Positive-Unlabeled (nnPU) loss.

    Parameters
    ----------
    prior : float
        Class prior π — the estimated fraction of true positives in the
        *unlabeled* set.  E.g. if ~30 % of ``alerted=0`` samples are
        actually attacks, set ``prior=0.30``.
    nnpu : bool
        When True (default) use the non-negative variant.  When False
        use the unbiased (uPU) estimator (can go negative, less stable).
    """

    def __init__(self, prior: float, nnpu: bool = True) -> None:
        super().__init__()
        if not 0.0 < prior < 1.0:
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior
        self.nnpu = nnpu
        self._warned_negative = False

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : Tensor [B]
            Raw (pre-sigmoid) model outputs.
        targets : Tensor [B]
            Labels — 1 for positive (confirmed), 0 for unlabeled.

        Returns
        -------
        loss : scalar Tensor
        """
        pos_mask = targets.bool()
        unl_mask = ~pos_mask
        n_pos = pos_mask.float().sum().clamp_min(1.0)
        n_unl = unl_mask.float().sum().clamp_min(1.0)

        # Per-sample sigmoid losses
        # ℓ(f, 1) = -log σ(f)   = softplus(-f)
        # ℓ(f, 0) = -log(1-σ(f)) = softplus(f)
        loss_pos_as_pos = F.softplus(-logits)   # ℓ(f(x), 1)
        loss_as_neg     = F.softplus(logits)    # ℓ(f(x), 0)

        # π · E_P[ ℓ(f(x), 1) ]
        positive_risk = self.prior * loss_pos_as_pos[pos_mask].sum() / n_pos

        # E_U[ ℓ(f(x), 0) ]
        unlabeled_neg_risk = loss_as_neg[unl_mask].sum() / n_unl

        # π · E_P[ ℓ(f(x), 0) ]  — correction term
        positive_neg_risk = self.prior * loss_as_neg[pos_mask].sum() / n_pos

        # Estimated negative risk
        negative_risk = unlabeled_neg_risk - positive_neg_risk

        if self.nnpu and negative_risk.item() < 0:
            if not self._warned_negative:
                logger.debug(
                    "nnPU: negative risk clamped to 0 (%.4f). "
                    "This is normal during early training.",
                    negative_risk.item(),
                )
                self._warned_negative = True
            # Use the gradient from the negative_risk term with a stop so
            # the model still gets a signal to push unlabeled outputs up,
            # but the total loss doesn't go below positive_risk.
            loss = positive_risk - negative_risk.detach() + negative_risk
        else:
            loss = positive_risk + negative_risk

        return loss
