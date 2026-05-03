"""
ByteTCN-FusionNet with prefix pooling.

Standalone components (ByteEncoder, TelemetryEncoder, FiLMGating,
ByteTCNFusionNet) are usable directly for advanced pipelines that
supply dual buffers and telemetry.

``ByteTCNClassifier`` is the registered wrapper that adapts the model
to the standard single-buffer pipeline used by the rest of the framework
(input shape ``[B, 1, L]``, single binary logit output).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseClassifier, register_model


# ──────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────

@dataclass
class ByteTCNFusionNetConfig:
    # Byte / sequence
    vocab_size: int = 257            # 256 bytes + PAD
    pad_idx: int = 256
    byte_embed_dim: int = 32

    # TCN
    tcn_channels: int = 96
    tcn_kernel_size: int = 5
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    tcn_dropout: float = 0.10
    norm_groups: int = 8

    # Prefix pooling
    prefix_lengths: Tuple[int, ...] = (128, 256, 512)

    # Encoded vector sizes
    buffer_proj_dim: int = 192
    telemetry_hidden_dim: int = 128
    fusion_hidden_dim: int = 256

    # Telemetry
    numeric_dim: int = 0
    categorical_cardinalities: Optional[Dict[str, int]] = None
    categorical_embed_dim: int = 8

    # Model behaviour
    share_byte_encoder: bool = False
    use_film: bool = True

    # Heads
    num_classes: Optional[int] = None  # optional multi-class auxiliary head

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ByteTCNFusionNetConfig":
        """Build config from a plain dictionary (e.g. JSON config section)."""
        kwargs: Dict[str, Any] = {}
        for k, v in d.items():
            if k == "tcn_dilations":
                kwargs[k] = tuple(v) if isinstance(v, list) else v
            elif k == "prefix_lengths":
                kwargs[k] = tuple(v) if isinstance(v, list) else v
            elif k == "categorical_cardinalities":
                kwargs[k] = v if isinstance(v, dict) else None
            else:
                kwargs[k] = v
        return cls(**kwargs)


# ──────────────────────────────────────────────
# Building blocks (shared blocks live in models.blocks)
# ──────────────────────────────────────────────

from .blocks import (  # noqa: E402
    DepthwiseSeparableConv1d,
    PrefixPooling,
    ResTCNBlock,
    masked_mean as _masked_mean,
    masked_max as _masked_max,
    make_prefix_mask as _make_prefix_mask,
)


class ByteEncoder(nn.Module):
    """bytes → embedding → stem conv → ResTCN stack → prefix pooling → projection."""

    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.byte_embed_dim, padding_idx=cfg.pad_idx)
        self.stem = nn.Conv1d(cfg.byte_embed_dim, cfg.tcn_channels, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList([
            ResTCNBlock(cfg.tcn_channels, cfg.tcn_kernel_size, d,
                        cfg.tcn_dropout, cfg.norm_groups)
            for d in cfg.tcn_dilations
        ])
        self.pool = PrefixPooling(cfg.prefix_lengths)
        pooled_dim = cfg.tcn_channels * 2 * (1 + len(cfg.prefix_lengths))
        self.proj = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, cfg.buffer_proj_dim),
            nn.SiLU(),
            nn.Dropout(cfg.tcn_dropout),
        )

    def forward(self, byte_ids: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if valid_mask is None:
            valid_mask = byte_ids.ne(self.cfg.pad_idx)
        x = self.embedding(byte_ids).transpose(1, 2)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.proj(self.pool(x, valid_mask))


class TelemetryEncoder(nn.Module):
    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        card = cfg.categorical_cardinalities or {}
        self.cat_names = list(card.keys())
        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinality, cfg.categorical_embed_dim)
            for name, cardinality in card.items()
        })
        cat_total_dim = len(self.cat_names) * cfg.categorical_embed_dim
        in_dim = cat_total_dim + cfg.numeric_dim
        if in_dim == 0:
            self.encoder = None
            self.out_dim = 0
        else:
            self.encoder = nn.Sequential(
                nn.Linear(in_dim, cfg.telemetry_hidden_dim), nn.SiLU(), nn.Dropout(0.10),
                nn.Linear(cfg.telemetry_hidden_dim, cfg.telemetry_hidden_dim), nn.SiLU(),
            )
            self.out_dim = cfg.telemetry_hidden_dim

    def forward(self, cat_features: Optional[Dict[str, torch.Tensor]] = None,
                num_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.encoder is None:
            if num_features is not None:
                return torch.zeros(num_features.size(0), 0, device=num_features.device)
            if cat_features:
                first = next(iter(cat_features.values()))
                return torch.zeros(first.size(0), 0, device=first.device)
            raise ValueError("TelemetryEncoder received no features and has zero input dims")
        parts: List[torch.Tensor] = []
        for name in self.cat_names:
            parts.append(self.cat_embeddings[name](cat_features[name].long()))
        if self.cfg.numeric_dim > 0:
            parts.append(num_features)
        return self.encoder(torch.cat(parts, dim=1))


class FiLMGating(nn.Module):
    """Telemetry-conditioned affine modulation: out = x * (1 + γ) + β."""

    def __init__(self, cond_dim: int, target_dim: int) -> None:
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, target_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.size(1) == 0:
            return x
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=1)
        return x * (1.0 + gamma) + beta


# ──────────────────────────────────────────────
# Full multi-head model (for advanced pipelines)
# ──────────────────────────────────────────────

class ByteTCNFusionNet(nn.Module):
    """
    Dual-buffer TCN with optional telemetry and FiLM gating.
    Returns a dict of logits / probabilities.
    """

    def __init__(self, cfg: ByteTCNFusionNetConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.header_encoder = ByteEncoder(cfg)
        self.body_encoder = self.header_encoder if cfg.share_byte_encoder else ByteEncoder(cfg)
        self.telemetry_encoder = TelemetryEncoder(cfg)

        if cfg.use_film and self.telemetry_encoder.out_dim > 0:
            self.header_film = FiLMGating(self.telemetry_encoder.out_dim, cfg.buffer_proj_dim)
            self.body_film = FiLMGating(self.telemetry_encoder.out_dim, cfg.buffer_proj_dim)
        else:
            self.header_film = None
            self.body_film = None

        fusion_in_dim = cfg.buffer_proj_dim * 2 + self.telemetry_encoder.out_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, cfg.fusion_hidden_dim), nn.SiLU(), nn.Dropout(0.15),
            nn.Linear(cfg.fusion_hidden_dim, cfg.fusion_hidden_dim), nn.SiLU(), nn.Dropout(0.10),
        )
        self.risk_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.snort_alert_head = nn.Linear(cfg.fusion_hidden_dim, 1)
        self.class_head = (
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_classes)
            if cfg.num_classes is not None and cfg.num_classes > 1
            else None
        )

    def forward(
        self,
        header_bytes: torch.Tensor,
        body_bytes: torch.Tensor,
        *,
        header_mask: Optional[torch.Tensor] = None,
        body_mask: Optional[torch.Tensor] = None,
        cat_features: Optional[Dict[str, torch.Tensor]] = None,
        num_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        header_vec = self.header_encoder(header_bytes, valid_mask=header_mask)
        body_vec = self.body_encoder(body_bytes, valid_mask=body_mask)
        if self.telemetry_encoder.out_dim > 0:
            telemetry_vec = self.telemetry_encoder(cat_features=cat_features, num_features=num_features)
        else:
            telemetry_vec = torch.zeros(header_bytes.size(0), 0, device=header_bytes.device)

        if self.header_film is not None:
            header_vec = self.header_film(header_vec, telemetry_vec)
        if self.body_film is not None:
            body_vec = self.body_film(body_vec, telemetry_vec)

        fused = self.fusion(torch.cat([header_vec, body_vec, telemetry_vec], dim=1))

        risk_logit = self.risk_head(fused).squeeze(-1)
        snort_logit = self.snort_alert_head(fused).squeeze(-1)

        out: Dict[str, torch.Tensor] = {
            "risk_logit": risk_logit,
            "risk_prob": torch.sigmoid(risk_logit),
            "snort_alert_logit": snort_logit,
            "snort_alert_prob": torch.sigmoid(snort_logit),
            "fused": fused,
            "header_vec": header_vec,
            "body_vec": body_vec,
            "telemetry_vec": telemetry_vec,
        }
        if self.class_head is not None:
            out["class_logits"] = self.class_head(fused)
        return out


# ──────────────────────────────────────────────
# Multi-task loss helper
# ──────────────────────────────────────────────

def compute_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    *,
    risk_target: torch.Tensor,
    snort_alert_target: Optional[torch.Tensor] = None,
    class_target: Optional[torch.Tensor] = None,
    risk_weight: float = 1.0,
    snort_weight: float = 0.25,
    class_weight: float = 0.25,
    pos_weight_risk: Optional[torch.Tensor] = None,
    pos_weight_snort: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = torch.tensor(0.0, device=outputs["risk_logit"].device)
    metrics: Dict[str, float] = {}

    risk_loss = F.binary_cross_entropy_with_logits(
        outputs["risk_logit"], risk_target.float(), pos_weight=pos_weight_risk,
    )
    total = total + risk_weight * risk_loss
    metrics["risk_loss"] = float(risk_loss.detach())

    if snort_alert_target is not None:
        snort_loss = F.binary_cross_entropy_with_logits(
            outputs["snort_alert_logit"], snort_alert_target.float(),
            pos_weight=pos_weight_snort,
        )
        total = total + snort_weight * snort_loss
        metrics["snort_alert_loss"] = float(snort_loss.detach())

    if class_target is not None and "class_logits" in outputs:
        class_loss = F.cross_entropy(outputs["class_logits"], class_target.long())
        total = total + class_weight * class_loss
        metrics["class_loss"] = float(class_loss.detach())

    metrics["total_loss"] = float(total.detach())
    return total, metrics


# ──────────────────────────────────────────────
# Framework-compatible wrapper
# ──────────────────────────────────────────────


@register_model("tcn")
class ByteTCNClassifier(BaseClassifier):
    """
    Dict-batch adapter for ``ByteTCNFusionNet``.

    Accepts a ``Dict[str, Tensor]`` batch with pre-split ``header_ids``
    and ``body_ids`` (produced by the unified data pipeline), and feeds
    them into the dual-buffer TCN.  Returns a ``[B]`` risk logit.
    """

    output_mode = "binary"

    def __init__(
        self,
        cfg: ByteTCNFusionNetConfig,
    ) -> None:
        super().__init__()
        self.tcn = ByteTCNFusionNet(cfg)
        self.cfg = cfg

    # ── forward ─────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch: Dict with at least ``header_ids``, ``body_ids`` (long [B, L]),
               and optionally ``header_mask``, ``body_mask`` (bool [B, L]).
        returns: [B] risk logit
        """
        header_ids = batch["header_ids"]
        body_ids = batch["body_ids"]
        header_mask = batch.get("header_mask")
        body_mask = batch.get("body_mask")

        outputs = self.tcn(
            header_ids, body_ids,
            header_mask=header_mask, body_mask=body_mask,
        )
        return outputs["risk_logit"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ByteTCNClassifier":
        model_cfg = config.get("model", {})

        # Collect TCN-specific keys, falling back to defaults
        tcn_keys = {
            "vocab_size", "pad_idx", "byte_embed_dim",
            "tcn_channels", "tcn_kernel_size", "tcn_dilations", "tcn_dropout",
            "norm_groups", "prefix_lengths", "buffer_proj_dim",
            "telemetry_hidden_dim", "fusion_hidden_dim",
            "numeric_dim", "categorical_cardinalities", "categorical_embed_dim",
            "share_byte_encoder", "use_film", "num_classes",
        }
        tcn_dict = {k: v for k, v in model_cfg.items() if k in tcn_keys}
        cfg = ByteTCNFusionNetConfig.from_dict(tcn_dict)

        return cls(cfg=cfg)
