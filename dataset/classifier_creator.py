import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Type Aliases
TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Metrics = Dict[str, float]


def set_global_seed(seed: int) -> None:
    """Sets seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")


class BufferDataset(Dataset):
    """Simple Tensor Dataset wrapper."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Size mismatch: features {features.shape[0]} != labels {labels.shape[0]}"
            )
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> TensorPair:
        return self.features[idx], self.labels[idx]


class DataUtils:
    """Static utility methods for data processing."""

    @staticmethod
    def buffer_to_fixed_array(buf: str, fixed_len: int) -> np.ndarray:
        """Converts a string buffer to a normalized float array of fixed length."""
        codes = [ord(c) for c in buf]
        arr = np.zeros(fixed_len, dtype=np.float32)
        n = min(len(codes), fixed_len)
        arr[:n] = codes[:n]
        arr /= 255.0  # Normalize
        return arr

    @staticmethod
    def ensure_path_list(maybe_path: Union[str, List[str], None]) -> List[Path]:
        """Normalizes config paths to a list of Path objects."""
        if maybe_path is None:
            return []
        if isinstance(maybe_path, str):
            return [Path(maybe_path)]
        if isinstance(maybe_path, list):
            return [Path(p) for p in maybe_path]
        raise TypeError(f"Invalid path type: {type(maybe_path)}")

    @staticmethod
    def resolve_device(arg_device: Optional[str]) -> torch.device:
        """Resolves the computation device (CUDA, MPS, or CPU)."""
        if arg_device:
            if arg_device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Fallback may occur.")
            return torch.device(arg_device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


class DatasetBuilder:
    """Handles loading JSON datasets, processing features, and splitting data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config.get("data", {})
        self.buffer_field = self.data_cfg.get("buffer_field", "buffer")
        self.fixed_len = int(self.data_cfg.get("fixed_len", 1024))
        self.seed = int(config.get("seed", 42))

    def build_datasets(self) -> Dict[str, BufferDataset]:
        """Builds train, val, and test datasets based on configuration."""
        benign_paths_cfg = self.config.get("benign_paths", {})
        attack_paths_cfg = self.config.get("attack_paths", {})
        attack_percent = float(self.config.get("attack_percent", 0.5))
        
        sampling_cfg = self.config.get("sampling", {})
        with_replacement = bool(sampling_cfg.get("with_replacement", False))
        
        # Offsets to ensure different splits get different random seeds
        split_seed_offsets = {"train": 11, "val": 22, "test": 33}
        datasets: Dict[str, BufferDataset] = {}

        for split in ["train", "val", "test"]:
            benign_paths = DataUtils.ensure_path_list(benign_paths_cfg.get(split))
            attack_paths = DataUtils.ensure_path_list(attack_paths_cfg.get(split))

            # Load all data for this split
            x_attack, y_train_attack, y_split_attack = self._load_group(
                attack_paths, default_label=1, is_test_split=(split == "test")
            )
            x_benign, y_train_benign, y_split_benign = self._load_group(
                benign_paths, default_label=0, is_test_split=(split == "test")
            )

            # Concatenate all
            x_all = torch.cat([x_attack, x_benign], dim=0)
            y_train_all = torch.cat([y_train_attack, y_train_benign], dim=0)
            y_split_all = torch.cat([y_split_attack, y_split_benign], dim=0)

            if x_all.numel() == 0:
                logger.warning(f"Split '{split}' is empty.")
                datasets[split] = BufferDataset(x_all, y_train_all)
                continue

            if split == "test":
                # For test set, do not mix/resample. Use all data as is.
                datasets[split] = BufferDataset(x_all, y_train_all)
                self._log_split_stats(split, y_train_all)
                continue

            # Split based on 'is_attack' (y_split), train on 'alerted' (y_train)
            pos_mask = y_split_all == 1
            neg_mask = y_split_all == 0

            # Separation
            pos_data = (x_all[pos_mask], y_train_all[pos_mask])
            neg_data = (x_all[neg_mask], y_train_all[neg_mask])

            # Mixing/Sampling
            current_seed = self.seed + split_seed_offsets.get(split, 0)
            mixed_x, mixed_y = self._mix_split(
                pos_data,
                neg_data,
                attack_percent,
                with_replacement,
                current_seed,
            )

            datasets[split] = BufferDataset(mixed_x, mixed_y)
            self._log_split_stats(split, mixed_y)

        return datasets

    def _load_group(self, paths: List[Path], default_label: int, is_test_split: bool = False) -> TensorTriple:
        """Loads a group of files (e.g., all benign files) and concatenates them."""
        features_list, train_labels_list, split_labels_list = [], [], []

        for p in paths:
            x, y_t, y_s = self._load_json_file(p, default_label, is_test_split)
            if x.numel() > 0:
                features_list.append(x)
                train_labels_list.append(y_t)
                split_labels_list.append(y_s)

        if not features_list:
            return (
                torch.empty(0, self.fixed_len, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        
        return (
            torch.cat(features_list, dim=0),
            torch.cat(train_labels_list, dim=0),
            torch.cat(split_labels_list, dim=0),
        )

    def _load_json_file(self, path: Path, default_label: int, is_test_split: bool = False) -> TensorTriple:
        """Parses a single JSON file into tensors."""
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return self._empty_triple()

        dataset = obj.get("dataset", [])
        if not isinstance(dataset, list):
            logger.error(f"File {path} invalid: 'dataset' is not a list")
            return self._empty_triple()

        features, train_labels, split_labels = [], [], []

        for rec in dataset:
            if not isinstance(rec, dict):
                continue
            
            buf = rec.get(self.buffer_field)
            if not isinstance(buf, str):
                continue

            # Determine Labels
            # 'alerted' -> User provided training label
            # 'is_attack' -> Ground truth for splitting (sampling balance)
            
            if is_test_split:
                 # In TEST split, we rely entirely on 'is_attack' (the ground truth)
                 # for both classification evaluation and splitting logic.
                is_attack_val = 1 if int(rec.get("is_attack", default_label)) == 1 else 0
                t_label = is_attack_val
                s_label = is_attack_val
            else:
                t_label = (
                    1 if int(rec.get("alerted", default_label)) == 1 else 0
                )  # Train label
                s_label = (
                    1 if int(rec.get("is_attack", default_label)) == 1 else 0
                )  # Split label

            arr = DataUtils.buffer_to_fixed_array(buf, self.fixed_len)
            features.append(arr)
            train_labels.append(t_label)
            split_labels.append(s_label)

        if not features:
            return self._empty_triple()

        return (
            torch.tensor(np.stack(features, axis=0), dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.long),
            torch.tensor(split_labels, dtype=torch.long),
        )

    def _empty_triple(self) -> TensorTriple:
        return (
            torch.empty(0, self.fixed_len, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

    def _mix_split(
        self,
        pos_data: TensorPair,
        neg_data: TensorPair,
        attack_percent: float,
        with_replacement: bool,
        seed: int,
    ) -> TensorPair:
        """Resamples positive and negative data to match start percentages."""
        pos_x, pos_y = pos_data
        neg_x, neg_y = neg_data
        n_pos, n_neg = pos_x.shape[0], neg_x.shape[0]

        n_pos_sample, n_neg_sample = self._compute_sample_counts(
            n_pos, n_neg, attack_percent, with_replacement
        )

        if n_pos_sample == 0 and n_neg_sample == 0:
            return torch.empty(0, self.fixed_len, dtype=torch.float32), torch.empty(0, dtype=torch.long)

        rng = np.random.RandomState(seed)

        sampled_pos_x, sampled_pos_y = self._sample_indices(
            pos_x, pos_y, n_pos_sample, with_replacement, rng
        )
        sampled_neg_x, sampled_neg_y = self._sample_indices(
            neg_x, neg_y, n_neg_sample, with_replacement, rng
        )

        all_x = torch.cat([sampled_pos_x, sampled_neg_x], dim=0)
        all_y = torch.cat([sampled_pos_y, sampled_neg_y], dim=0)

        # Shuffle combined
        perm = torch.from_numpy(rng.permutation(all_x.shape[0])).long()
        return all_x[perm], all_y[perm]

    def _sample_indices(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_sample: int,
        replace: bool,
        rng: np.random.RandomState,
    ) -> TensorPair:
        n_available = x.shape[0]
        if n_available == 0 or n_sample == 0:
            return torch.empty(0, self.fixed_len, dtype=torch.float32), torch.empty(0, dtype=torch.long)

        if replace or n_sample > n_available:
            indices = rng.randint(0, n_available, size=n_sample)
        else:
            indices = rng.choice(n_available, size=n_sample, replace=False)
        
        return x[indices], y[indices]

    def _compute_sample_counts(
        self, n_pos: int, n_neg: int, attack_percent: float, with_replacement: bool
    ) -> Tuple[int, int]:
        p = float(attack_percent)
        if p <= 0.0:
            return 0, n_neg
        if p >= 1.0:
            return n_pos, 0

        if with_replacement:
            total = n_pos + n_neg
            target_pos = int(round(p * total))
            target_neg = total - target_pos
        else:
            # Maximimize dataset size while respecting ratio without replacement
            max_pos = n_pos / p if p > 0 else float("inf")
            max_neg = n_neg / (1.0 - p) if p < 1.0 else float("inf")
            total = int(min(max_pos, max_neg))
            
            target_pos = min(int(round(p * total)), n_pos)
            target_neg = min(total - target_pos, n_neg)

        return max(target_pos, 0), max(target_neg, 0)

    def _log_split_stats(self, name: str, labels: torch.Tensor) -> None:
        total = labels.shape[0]
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        logger.info(f"Split {name}: Total={total}, Pos={pos}, Neg={neg}")


class Conv1DClassifier(nn.Module):
    """1D Convolutional Neural Network for sequence classification."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_channels: List[int],
        kernel_sizes: List[int],
        dropout: float,
        input_length: int,
    ) -> None:
        super().__init__()
        assert len(conv_channels) == len(kernel_sizes)

        layers: List[nn.Module] = []
        prev_c = in_channels
        length = input_length

        for out_c, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv1d(prev_c, out_c, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_c = out_c
            length = max(length // 2, 1)

        self.features = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        
        # Calculate linear layer size
        self.classifier = nn.Linear(prev_c * length, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


class MetricUtils:
    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
        """Computes Accuracy, Precision, Recall, and F1-Score."""
        assert y_true.shape == y_pred.shape

        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))

        eps = 1e-12
        return {
            "accuracy": (tp + tn) / max(tp + tn + fp + fn, eps),
            "precision": tp / max(tp + fp, eps),
            "recall": tp / max(tp + fn, eps),
            "f1": 2.0 * tp / max(2.0 * tp + fp + fn, eps),
        }


class Trainer:
    """Manages the training loop and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> Tuple[float, Metrics]:
        self.model.train(train)
        
        all_losses: List[float] = []
        all_true: List[int] = []
        all_pred: List[int] = []

        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb = xb.to(self.device).unsqueeze(1)
                yb = yb.to(self.device)

                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                all_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1)
                all_true.extend(yb.tolist())
                all_pred.extend(preds.tolist())

        avg_loss = float(np.mean(all_losses)) if all_losses else 0.0
        
        if not all_true:
            return avg_loss, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        metrics = MetricUtils.compute_binary_metrics(
            np.array(all_true), np.array(all_pred)
        )
        metrics["loss"] = avg_loss
        return avg_loss, metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        best_metric_name: str,
        out_dir: Path,
    ) -> Tuple[Dict[str, List[Metrics]], Metrics]:
        
        best_metric = -float("inf")
        best_state: Optional[Dict[str, Any]] = None
        history: Dict[str, List[Metrics]] = {"train": [], "val": []}

        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self._run_epoch(train_loader, train=True)
            _, val_metrics = self._run_epoch(val_loader, train=False)

            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            metric_val = val_metrics.get(best_metric_name, 0.0)
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                }

            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

        if best_state:
            torch.save(best_state, out_dir / "model_best.pt")
            logger.info(f"Best model saved with {best_metric_name}={best_metric:.4f}")
            return history, best_state["val_metrics"]

        return history, {}

    def evaluate(self, loader: DataLoader) -> Metrics:
        _, metrics = self._run_epoch(loader, train=False)
        return metrics


class LSTMClassifier(nn.Module):
    """LSTM-based classifier for sequence classification."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1), num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels=1, seq_len) -> (batch, seq_len, channels=1)
        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            # Concatenate last forward and last backward states from the last layer
            final_h = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            final_h = h_n[-1]

        return self.fc(final_h)


def build_model(config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})

    model_type = model_cfg.get("type", "cnn").lower()

    if model_type == "lstm":
        logger.info("Building LSTM model...")
        return LSTMClassifier(
            input_size=int(model_cfg.get("in_channels", 1)),
            hidden_size=int(model_cfg.get("hidden_size", 64)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            num_classes=int(model_cfg.get("num_classes", 2)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            bidirectional=bool(model_cfg.get("bidirectional", False)),
        )
    
    logger.info("Building CNN model...")
    return Conv1DClassifier(
        in_channels=int(model_cfg.get("in_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 2)),
        conv_channels=model_cfg.get("conv_channels", [16, 32, 64]),
        kernel_sizes=model_cfg.get("kernel_sizes", [7, 5, 3]),
        dropout=float(model_cfg.get("dropout", 0.0)),
        input_length=int(data_cfg.get("fixed_len", 1024)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train 1D-CNN Classifier")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--device", help="Override device (cpu, cuda, mps)")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run single batch only")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    # Overrides
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    
    set_global_seed(int(config.get("seed", 42)))
    device = DataUtils.resolve_device(args.device)

    # Build Data
    builder = DatasetBuilder(config)
    datasets = builder.build_datasets()

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 128))
    
    # Loaders
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0
        )
        for split, ds in datasets.items()
    }

    model = build_model(config)

    if args.dry_run:
        logger.info("Dry run initiated.")
        if len(loaders["train"]) > 0:
            xb, _ = next(iter(loaders["train"]))
            xb = xb.to(device).unsqueeze(1)
            model = model.to(device)
            with torch.no_grad():
                out = model(xb)
            logger.info(f"Forward pass successful. Shape: {out.shape}")
        return 0

    # Setup Artifacts
    artifacts_cfg = config.get("artifacts", {})
    out_dir = Path(artifacts_cfg.get("out_dir", "./artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    history, best_metrics = trainer.train(
        loaders["train"],
        loaders["val"],
        epochs=int(training_cfg.get("epochs", 10)),
        best_metric_name=artifacts_cfg.get("best_metric", "f1"),
        out_dir=out_dir,
    )

    # Save Last
    torch.save(
        {"model_state_dict": model.state_dict()}, 
        out_dir / "model_last.pt"
    )

    # Test
    test_metrics = trainer.evaluate(loaders["test"])
    logger.info("Test Metrics:")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Dump Results
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_val_metrics": best_metrics,
                "test_metrics": test_metrics,
            },
            f,
            indent=2,
            sort_keys=True
        )
    
    with (out_dir / "config_used.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
