import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from .data import DatasetBuilder
from .models import build_model
from .trainer import Trainer
from .local_types import Metrics

logger = logging.getLogger(__name__)


class ClassifierPipeline:
    """High-level training and evaluation pipeline."""

    def __init__(self, config: Dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

    def build_loaders(self) -> Dict[str, DataLoader]:
        builder = DatasetBuilder(self.config)
        datasets = builder.build_datasets()

        training_cfg = self.config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 128))

        return {
            split: DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0,
            )
            for split, ds in datasets.items()
        }

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

        self._save_test_samples(out_dir, model, loaders["test"], threshold=threshold)

        self._dump_results(out_dir, history, best_metrics, test_metrics)

        return history, best_metrics, test_metrics

    def _run_dry(self, loader: Optional[DataLoader], model: torch.nn.Module) -> None:
        logger.info("Dry run initiated.")
        if not loader or len(loader) == 0:
            logger.warning("Dry run skipped: empty train loader.")
            return
        xb, _ = next(iter(loader))
        xb = xb.to(self.device).unsqueeze(1)
        model = model.to(self.device)
        with torch.no_grad():
            out = model(xb)
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

    def _save_test_samples(
        self, out_dir: Path, model: torch.nn.Module, loader: DataLoader,
        threshold: float = 0.5,
    ) -> None:
        testing_cfg = self.config.get("testing", {})
        max_samples = int(testing_cfg.get("log_samples", 200))

        if max_samples <= 0:
            return

        model.eval()
        logged = 0
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    xb, yb, rb = batch[0], batch[1], batch[2]
                else:
                    xb, yb = batch
                    rb = ["" for _ in range(len(yb))]

                xb = xb.to(self.device).unsqueeze(1)
                yb = yb.to(self.device)
                logits = model(xb)
                if logits.dim() == 1:
                    probs = torch.sigmoid(logits)
                    preds = (probs >= threshold).long()
                else:
                    preds = torch.argmax(logits, dim=1)

                for i in range(xb.shape[0]):
                    if logged >= max_samples:
                        break
                    expected = int(yb[i].item())
                    received = int(preds[i].item())
                    raw_buffer = rb[i]
                    if isinstance(raw_buffer, torch.Tensor):
                        raw_buffer = raw_buffer.item()
                    rows.append(
                        {
                            "expected": expected,
                            "received": received,
                            "raw_buffer": raw_buffer,
                        }
                    )
                    logged += 1

        out_path = out_dir / "test_samples.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=True)
        logger.info("Saved %s test samples to %s", len(rows), out_path)
