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

from ..data.common import to_device
from ..data import TwoWayDatasetBuilder
from ..models import build_model
from ..training.twoway import TwoWayTrainer, eval_on_loader_at_threshold
from ..training.calibration import BaseCalibrator, IsotonicCalibrator, PriorCorrectionCalibrator, pick_best_calibrator

logger = logging.getLogger(__name__)


class TwoWayPipeline:
    """High-level pipeline for the 2-Way ByteTCN model."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def run(
        self,
        stop_flag: Optional[Callable[[], bool]] = None,
        pretrained_path: Optional[str] = None,
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

        # 3. Train — optionally skip Stage 1 by loading a pretrained checkpoint
        trainer = TwoWayTrainer(model, self.device, training_cfg)

        if pretrained_path:
            pt_path = Path(pretrained_path)
            if not pt_path.exists():
                raise FileNotFoundError(f"Pretrained checkpoint not found: {pt_path}")
            trainer.load_pretrained(pt_path)
            pretrain_stats = trainer.evaluate(loaders["val"])
            logger.info("Skipping Stage 1; loaded pretrained weights.")
        else:
            pretrain_stats = trainer.pretrain(
                loaders["train_u"], loaders["val"], out_dir, stop_flag=stop_flag,
            )

        best_val, best_path = trainer.train_pu(
            loaders["train_p"], loaders["train_u"], loaders["val"],
            out_dir, stop_flag=stop_flag,
        )

        # 4. Load best checkpoint and calibrate
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=self.device, weights_only=True)
            model.backbone.load_state_dict(ckpt["backbone"])
            model.heads.load_state_dict(ckpt["heads"])

        # Collect val logits for calibration
        val_logits, val_labels = self._collect_logits(model, loaders["val"])
        pi_train = float(training_cfg.get("pi_p", training_cfg.get("pu_prior", 0.10)))
        calibrator = IsotonicCalibrator()
        calibrator.fit(val_logits, val_labels)
        calibrator.save(out_dir / "calibrator.json")

        # Evaluate test with calibrated threshold (0.5 on calibrated scores)
        test_stats = self._eval_calibrated(
            model, loaders["test"], calibrator, best_val,
        )
        logger.info("Test Metrics (calibrated, threshold=0.5):")
        for k, v in test_stats.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %s", k, v)

        # 5. Save per-sample prognosis (with calibrated scores)
        self._save_sample_prognosis(out_dir, model, loaders["val"], "val", calibrator)
        self._save_sample_prognosis(out_dir, model, loaders["test"], "test", calibrator)

        # 6. Save artifacts
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

    @torch.no_grad()
    def _collect_logits(
        self, model: Any, loader: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over a loader, return (logits, labels) on CPU."""
        model.backbone.eval()
        model.heads.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for batch in loader:
            batch = to_device(batch, self.device)
            z = model.backbone(batch)
            out = model.heads(z)
            all_logits.append(out["risk_logit"].cpu())
            all_labels.append(batch["is_attack"].cpu())
        return torch.cat(all_logits), torch.cat(all_labels)

    @torch.no_grad()
    def _eval_calibrated(
        self,
        model: Any,
        loader: Any,
        calibrator: BaseCalibrator,
        best_val: Dict[str, float],
    ) -> Dict[str, float]:
        """Evaluate using calibrated scores at threshold=0.5."""
        from ..training.metrics import pr_curve_best_f1

        logits, labels = self._collect_logits(model, loader)
        cal_scores = calibrator.transform(logits)
        raw_scores = torch.sigmoid(logits)

        # Fixed threshold at 0.5 on calibrated scores
        preds = (cal_scores >= 0.5).float()
        tp = (preds * labels).sum()
        fp = (preds * (1 - labels)).sum()
        fn = ((1 - preds) * labels).sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)

        # PR-AUC on raw scores for reference
        sweep = pr_curve_best_f1(raw_scores, labels)

        return {
            "threshold": 0.5,
            "f1": float(f1.item()),
            "precision": float(precision.item()),
            "recall": float(recall.item()),
            "pr_auc": sweep["pr_auc"],
            "best_f1_sweep": sweep["best_f1"],
            "best_threshold_sweep": sweep["best_threshold"],
            "calibrator": calibrator.name,
            "val_threshold_raw": best_val.get("best_threshold", 0.5),
        }

    @torch.no_grad()
    def _save_sample_prognosis(
        self, out_dir: Path, model: Any, loader: Any, split: str,
        calibrator: Optional[BaseCalibrator] = None,
    ) -> None:
        """Dump per-sample raw logits, scores, and ground-truth labels."""
        model.backbone.eval()
        model.heads.eval()
        rows: list[dict[str, Any]] = []

        for batch in loader:
            batch = to_device(batch, self.device)
            z = model.backbone(batch)
            out = model.heads(z)
            logits = out["risk_logit"]
            scores = torch.sigmoid(logits)
            cal_scores = calibrator.transform(logits.cpu()).to(logits.device) if calibrator else scores
            n = logits.shape[0]

            is_attack = batch.get("is_attack")
            alerted = batch.get("alerted")

            for i in range(n):
                row: dict[str, Any] = {
                    "split": split,
                    "raw_logit": round(float(logits[i].item()), 4),
                    "raw_score": round(float(scores[i].item()), 4),
                    "calibrated_score": round(float(cal_scores[i].item()), 4),
                }
                if is_attack is not None:
                    row["is_attack"] = int(is_attack[i].item())
                if alerted is not None:
                    row["alerted"] = int(alerted[i].item())
                rows.append(row)

        out_path = out_dir / f"{split}_samples.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=True)
        logger.info("Saved %d %s samples to %s", len(rows), split, out_path)
        logger.info("Saved %d %s samples to %s", len(rows), split, out_path)

    # ── calibrate-only mode ─────────────────────────

    def calibrate(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a trained model, fit the best calibrator on val, evaluate on test.

        No training happens — purely inference + calibration.
        """
        # 1. Build data loaders
        builder = TwoWayDatasetBuilder(self.config)
        loaders = builder.build_loaders()

        artifacts_cfg = self.config.get("artifacts", {})
        out_dir = Path(artifacts_cfg.get("out_dir", "./artifacts"))
        out_dir.mkdir(parents=True, exist_ok=True)

        # 2. Build model & load checkpoint
        model = build_model(self.config)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.backbone.load_state_dict(ckpt["backbone"])
        model.heads.load_state_dict(ckpt["heads"])
        model.backbone.to(self.device)
        model.heads.to(self.device)
        logger.info("Loaded checkpoint: %s", checkpoint_path)

        # 3. Collect val logits and pick best calibrator
        val_logits, val_labels = self._collect_logits(model, loaders["val"])
        training_cfg = self.config.get("training", {})
        pi_train = float(training_cfg.get("pi_p", training_cfg.get("pu_prior", 0.10)))

        logger.info("Fitting calibrators...")
        calibrator = pick_best_calibrator(val_logits, val_labels, pi_train=pi_train)
        calibrator.save(out_dir / "calibrator.json")
        logger.info("Saved calibrator to %s", out_dir / "calibrator.json")

        # 4. Evaluate on test
        test_stats = self._eval_calibrated(model, loaders["test"], calibrator, {})
        logger.info("Test Metrics (calibrated, threshold=0.5):")
        for k, v in test_stats.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %s", k, v)

        # 5. Save per-sample prognosis
        self._save_sample_prognosis(out_dir, model, loaders["val"], "val", calibrator)
        self._save_sample_prognosis(out_dir, model, loaders["test"], "test", calibrator)

        return test_stats
