"""
2-Way ByteTCN backbone with multi-head outputs.

This model is designed for two-stage training: contrastive SSL pretraining
followed by nnPU multitask learning.  It operates on pre-split header/body
byte sequences provided by the data pipeline.

Unlike the standard ``ByteTCNClassifier``, this model does NOT conform to
the single-tensor ``[B,1,L]`` interface.  It requires ``TwoWayPipeline``.

Note
----
The shared NN building blocks (``DepthwiseSeparableConv1d``, ``ResTCNBlock``,
``PrefixPooling``, ``masked_*``) are provided by ``models.blocks``.
The loss functions (``nnpu_loss``, ``contrastive_nt_xent``) are provided by
``losses``.  Importing them from this module is deprecated.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import PrefixPooling, ResTCNBlock
from .base import BaseClassifier, register_model


# ─── Constants ─────────────────────────────────────────

PAD_IDX = 256
VOCAB_SIZE = 257


# ─── Byte encoder (kwargs-based for the 2-way pipeline) ──


class ByteEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        channels: int = 96,
        kernel: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        prefix_lengths: Tuple[int, ...] = (64, 128, 256),
        proj_dim: int = 192,
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_IDX)
        self.stem = nn.Conv1d(embed_dim, channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList(
            [ResTCNBlock(channels, kernel, d, dropout) for d in dilations]
        )
        self.pool = PrefixPooling(prefix_lengths)

        pooled_mul = 2 * (1 + len(prefix_lengths))
        pooled_dim = channels * pooled_mul
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, proj_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids).transpose(1, 2)  # [B, E, L]
        x = self.stem(x)                   # [B, C, L]
        for b in self.blocks:
            x = b(x)
        pooled = self.pool(x, mask)         # [B, pooled_dim]
        return self.proj(pooled)            # [B, proj_dim]


# ─── Backbone ──────────────────────────────────────────


class ByteTCNBackbone(nn.Module):
    """Returns a representation vector z for each sample."""

    def __init__(
        self,
        embed_dim: int = 32,
        channels: int = 96,
        kernel: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.1,
        prefix_lengths: Tuple[int, ...] = (64, 128, 256),
        proj_dim: int = 192,
        fusion_dim: int = 256,
    ) -> None:
        super().__init__()
        self.enc_h = ByteEncoder(embed_dim, channels, kernel, dilations,
                                 dropout, prefix_lengths, proj_dim)
        self.enc_b = ByteEncoder(embed_dim, channels, kernel, dilations,
                                 dropout, prefix_lengths, proj_dim)

        fusion_in = proj_dim * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, fusion_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim, fusion_dim),
            nn.SiLU(),
        )
        self.out_dim = fusion_dim

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        zh = self.enc_h(batch["header_ids"], batch["header_mask"])
        zb = self.enc_b(batch["body_ids"], batch["body_mask"])
        z = torch.cat([zh, zb], dim=1)
        return self.fusion(z)


# ─── Heads ─────────────────────────────────────────────


class Heads(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.risk = nn.Linear(in_dim, 1)
        self.alerted = nn.Linear(in_dim, 1)
        # projection head for contrastive SSL
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.SiLU(),
            nn.Linear(in_dim, 128),
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "risk_logit": self.risk(z).squeeze(-1),
            "alerted_logit": self.alerted(z).squeeze(-1),
            "proj": F.normalize(self.proj(z), dim=1),
        }


__all__ = [
    "PAD_IDX",
    "VOCAB_SIZE",
    "ByteEncoder",
    "ByteTCNBackbone",
    "ByteTCN2WayClassifier",
    "Heads",
]


# ─── Registered wrapper ───────────────────────────────


@register_model("tcn_2way")
class ByteTCN2WayClassifier(BaseClassifier):
    """Factory-compatible wrapper around ByteTCNBackbone + Heads.

    Exposes a single ``forward(batch) → risk_logit`` interface for
    inference, while providing ``.backbone`` and ``.heads`` attributes
    for two-stage training access.
    """

    output_mode = "binary"

    def __init__(self, backbone: ByteTCNBackbone, heads: Heads) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = heads

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backbone(batch)
        out = self.heads(z)
        return out["risk_logit"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ByteTCN2WayClassifier":
        model_cfg = config.get("model", {})
        backbone = ByteTCNBackbone(
            embed_dim=int(model_cfg.get("embed_dim", 32)),
            channels=int(model_cfg.get("channels", 96)),
            kernel=int(model_cfg.get("kernel", 5)),
            dilations=tuple(model_cfg.get("dilations", [1, 2, 4, 8])),
            dropout=float(model_cfg.get("dropout", 0.1)),
            prefix_lengths=tuple(model_cfg.get("prefix_lengths", [64, 128, 256])),
            proj_dim=int(model_cfg.get("proj_dim", 192)),
            fusion_dim=int(model_cfg.get("fusion_dim", 256)),
        )
        heads = Heads(backbone.out_dim)
        return cls(backbone, heads)
