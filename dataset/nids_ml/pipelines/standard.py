import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..data import DatasetBuilder
from ..models import build_model
from ..training.standard import Trainer, _is_dict_model
from ..local_types import Metrics

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """High-level training and evaluation pipeline."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def build_loaders(self) -> Dict[str, DataLoader]:
        builder = DatasetBuilder(self.config)
        return builder.build_loaders()

    def run(
        self,
        epochs_override: Optional[int] = None,
        dry_run: bool = False,
        stop_flag: Optional[Callable[[], bool]] = None,
    ) -> Tuple[Dict[str, Any], Metrics, Metrics]:
        loaders = self.build_loaders()
        model = build_model(self.config).to(self.device)

        if dry_run:
            self._run_dry(loader=loaders.get("train"), model=model)
            return {}, {}, {}

        artifacts_cfg = self.config.get("artifacts", {})
        out_dir = Path(artifacts_cfg.get("out_dir", "./artifacts"))
        out_dir.mkdir(parents=True, exist_ok=True)

        training_cfg = self.config.get("training", {})
        epochs = int(epochs_override or training_cfg.get("epochs", 10))

        model_cfg = self.config.get("model", {})
        class_weights = None
        weights_cfg = model_cfg.get("class_weights")
        if isinstance(weights_cfg, (list, tuple)):
            class_weights = torch.tensor(weights_cfg, dtype=torch.float32)

        output_mode = getattr(model, "output_mode", "multiclass")
        threshold = float(model_cfg.get("threshold", 0.5))

        pu_prior_cfg = training_cfg.get("pu_prior")
        pu_prior = float(pu_prior_cfg) if pu_prior_cfg is not None else None

        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
            class_weights=class_weights,
            output_mode=output_mode,
            threshold=threshold,
            patience=int(training_cfg.get("patience", 0)),
            lr_scheduler_cfg=training_cfg.get("lr_scheduler"),
            pu_prior=pu_prior,
        )

        history, best_metrics, last_state = trainer.train(
            loaders["train"],
            loaders["val"],
            epochs=epochs,
            best_metric_name=artifacts_cfg.get("best_metric", "f1"),
            out_dir=out_dir,
            stop_flag=stop_flag,
        )

        if last_state:
            torch.save(last_state, out_dir / "model_last.pt")
        else:
            torch.save({"model_state_dict": model.state_dict()}, out_dir / "model_last.pt")

        test_metrics = trainer.evaluate(loaders["test"])
        logger.info("Test Metrics:")
        for k, v in test_metrics.items():
            logger.info("  %s: %.4f", k, v)

        self._save_sample_prognosis(out_dir, model, loaders["val"], "val")
        self._save_sample_prognosis(out_dir, model, loaders["test"], "test")

        self._dump_results(out_dir, history, best_metrics, test_metrics)

        return history, best_metrics, test_metrics

    def _run_dry(self, loader: Optional[DataLoader], model: torch.nn.Module) -> None:
        logger.info("Dry run initiated.")
        if not loader or len(loader) == 0:
            logger.warning("Dry run skipped: empty train loader.")
            return
        batch = next(iter(loader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        model = model.to(self.device)
        dict_mode = _is_dict_model(model)
        with torch.no_grad():
            if dict_mode:
                out = model(batch)
            else:
                ids = torch.cat([batch["header_ids"], batch["body_ids"]], dim=1)
                out = model(ids)
        logger.info("Forward pass successful. Shape: %s", tuple(out.shape))

    def _dump_results(
        self,
        out_dir: Path,
        history: Dict[str, Any],
        best_metrics: Metrics,
        test_metrics: Metrics,
    ) -> None:
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "history": history,
                    "best_val_metrics": best_metrics,
                    "test_metrics": test_metrics,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, sort_keys=True)

    def _save_sample_prognosis(
        self, out_dir: Path, model: torch.nn.Module, loader: DataLoader,
        split: str,
    ) -> None:
        """Dump per-sample raw logits, scores, and ground-truth labels."""
        model.eval()
        dict_mode = _is_dict_model(model)
        rows: list[dict[str, Any]] = []

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    is_attack = batch.get("is_attack")
                    alerted = batch.get("alerted")
                    if dict_mode:
                        logits = model(batch)
                    else:
                        ids = torch.cat(
                            [batch["header_ids"], batch["body_ids"]], dim=1,
                        )
                        logits = model(ids)
                    n = batch["header_ids"].shape[0]
                else:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        xb, yb = batch[0], batch[1]
                    else:
                        xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = model(xb)
                    is_attack = yb
                    alerted = None
                    n = xb.shape[0]

                if logits.dim() == 1:
                    scores = torch.sigmoid(logits)
                else:
                    scores = torch.softmax(logits, dim=1)[:, 1]
                    logits = logits[:, 1] - logits[:, 0]

                for i in range(n):
                    row: dict[str, Any] = {
                        "split": split,
                        "raw_logit": round(float(logits[i].item()), 4),
                        "raw_score": round(float(scores[i].item()), 4),
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
