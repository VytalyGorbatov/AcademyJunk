"""
Shared neural-network building blocks used by all TCN-based models.

Centralising these blocks eliminates the duplication that previously existed
across ``models/tcn.py``, ``models/tcn_2way.py`` and the archived standalone
scripts.  All TCN variants in the package should import from this module
rather than defining local copies.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


# ─── Masked pooling utilities ───────────────────────────


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean over *dim* using a boolean *mask* (True = valid).

    Shape conventions are flexible: *mask* must be broadcastable to *x*.
    Fully-masked positions are protected by clamping the denominator to 1.
    """
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return (x * mask_f).sum(dim=dim) / denom


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Max over *dim* using a boolean *mask* (True = valid).

    Rows that are fully masked produce 0 instead of -inf.
    """
    x_masked = x.masked_fill(~mask, float("-inf"))
    out = x_masked.max(dim=dim).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def make_prefix_mask(base_mask: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """Restrict *base_mask* (``[B, L]`` bool) to the first *prefix_len* positions."""
    seq_len = base_mask.size(1)
    device = base_mask.device
    prefix_pos = torch.arange(seq_len, device=device).unsqueeze(0) < prefix_len
    return base_mask & prefix_pos


# ─── Convolutional building blocks ──────────────────────


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable 1-D convolution with dilation and post-dropout."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.dw = nn.Conv1d(
            channels, channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=channels, bias=False,
        )
        self.pw = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pw(self.dw(x)))


class ResTCNBlock(nn.Module):
    """Residual TCN block: pre-norm → SiLU → DSConv → pre-norm → SiLU → DSConv + residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        groups = min(norm_groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1

        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = DepthwiseSeparableConv1d(channels, kernel_size, dilation, dropout)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = DepthwiseSeparableConv1d(channels, kernel_size, 1, dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(x))
        out = self.conv1(out)
        out = self.act(self.norm2(out))
        out = self.conv2(out)
        return residual + out


# ─── Pooling ────────────────────────────────────────────


class PrefixPooling(nn.Module):
    """Global + prefix-windowed mean/max pooling.

    Output dim is ``2 * (1 + len(prefix_lengths)) * C``.
    """

    def __init__(self, prefix_lengths: Sequence[int]) -> None:
        super().__init__()
        self.prefix_lengths = tuple(prefix_lengths)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]   valid_mask: [B, L]
        mask_3d = valid_mask.unsqueeze(1)  # [B, 1, L]
        pooled = [
            masked_mean(x, mask_3d, dim=2),
            masked_max(x, mask_3d, dim=2),
        ]
        seq_len = x.size(-1)
        for k in self.prefix_lengths:
            p_mask = make_prefix_mask(valid_mask, min(k, seq_len)).unsqueeze(1)
            pooled.append(masked_mean(x, p_mask, dim=2))
            pooled.append(masked_max(x, p_mask, dim=2))
        return torch.cat(pooled, dim=1)


__all__ = [
    "masked_mean",
    "masked_max",
    "make_prefix_mask",
    "DepthwiseSeparableConv1d",
    "ResTCNBlock",
    "PrefixPooling",
]
