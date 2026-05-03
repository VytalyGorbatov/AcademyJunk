"""
Pipeline orchestration for the 2-Way ByteTCN.

Coordinates data loading, model construction, two-stage training,
evaluation, and artifact persistence.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from ..data import TwoWayDatasetBuilder
from ..models import build_model
from ..training.twoway import TwoWayTrainer

logger = logging.getLogger(__name__)


class TwoWayPipeline:
    """High-level pipeline for the 2-Way ByteTCN model."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float]]:
        # 1. Build data loaders
        builder = TwoWayDatasetBuilder(self.config)
        loaders = builder.build_loaders()

        artifacts_cfg = self.config.get("artifacts", {})
        out_dir = Path(artifacts_cfg.get("out_dir", "./artifacts"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. Build model via factory
        training_cfg = self.config.get("training", {})
        model = build_model(self.config)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("Model parameters: %s", f"{n_params:,}")

        # 3. Train — Stage 1 (pretraining) then Stage 2 (nnPU)
        trainer = TwoWayTrainer(model, self.device, training_cfg)

        pretrain_stats = trainer.pretrain(
            loaders["train_u"], loaders["val"], out_dir, stop_flag=stop_flag,
        )

        best_val, best_path = trainer.train_pu(
            loaders["train_p"], loaders["train_u"], loaders["val"],
            out_dir, stop_flag=stop_flag,
        )

        # 4. Evaluate on test with the best checkpoint
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=True)
            model.backbone.load_state_dict(ckpt["backbone"])
            model.heads.load_state_dict(ckpt["heads"])

        test_stats = trainer.evaluate(loaders["test"])
        logger.info("Test Metrics:")
        for k, v in test_stats.items():
            logger.info("  %s: %.4f", k, v)

        # 5. Save artifacts
        self._dump_results(out_dir, pretrain_stats, best_val, test_stats)

        return {}, best_val, test_stats

    # ── artifact persistence ────────────────────────

    def _dump_results(
        self,
        out_dir: Path,
        pretrain_stats: Dict[str, float],
        best_val: Dict[str, float],
        test_stats: Dict[str, float],
    ) -> None:
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "pretrain_val": pretrain_stats,
                    "best_val_metrics": best_val,
                    "test_metrics": test_stats,
                },
                f, indent=2, sort_keys=True,
            )

        with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)
