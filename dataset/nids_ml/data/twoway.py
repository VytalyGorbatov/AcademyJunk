"""TwoWayDatasetBuilder — builds P/U/val/test loaders for 2-way training."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import DataLoader

from .common import (
    SEP_BYTE,
    DataConfig2Way,
    TwoWayRecordDataset,
    twoway_collate_fn,
)

logger = logging.getLogger(__name__)


class TwoWayDatasetBuilder:
    """Builds P, U, val, and test loaders from existing JSON data files."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        data_dict = config.get("data", {})
        self.cfg = DataConfig2Way(
            fixed_len=int(data_dict.get("fixed_len", 1024)),
            fallback_header_len=int(data_dict.get("fallback_header_len", 512)),
            buffer_field=data_dict.get("buffer_field", "buffers"),
            sep_byte=int(data_dict.get("sep_byte", SEP_BYTE)),
        )

    # ── path helpers ────────────────────────────────

    @staticmethod
    def _ensure_path_list(val: Any) -> List[Path]:
        if val is None:
            return []
        if isinstance(val, str):
            return [Path(val)]
        if isinstance(val, list):
            return [Path(p) for p in val]
        raise TypeError(f"Invalid path type: {type(val)}")

    def _load_records_from_file(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        dataset = obj.get("dataset", [])
        if not isinstance(dataset, list):
            logger.error("File %s invalid: 'dataset' is not a list", path)
            return []
        return [rec for rec in dataset if isinstance(rec, dict) and self.cfg.buffer_field in rec]

    def _load_split_records(self, split: str) -> List[Dict[str, Any]]:
        benign = self._ensure_path_list(self.config.get("benign_paths", {}).get(split))
        attack = self._ensure_path_list(self.config.get("attack_paths", {}).get(split))
        records: List[Dict[str, Any]] = []
        for p in benign + attack:
            records.extend(self._load_records_from_file(p))
        return records

    # ── main entry ──────────────────────────────────

    def build_loaders(self) -> Dict[str, DataLoader]:
        training_cfg = self.config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 256))
        num_workers = int(training_cfg.get("num_workers", 0))

        train_records = self._load_split_records("train")
        val_records = self._load_split_records("val")
        test_records = self._load_split_records("test")

        # Split train into P (alerted=1) and U (alerted=0)
        p_records = [r for r in train_records if int(r.get("alerted", 0)) == 1]
        u_records = [r for r in train_records if int(r.get("alerted", 0)) == 0]

        logger.info(
            "Train P (alerted=1): %d, Train U (alerted=0): %d",
            len(p_records), len(u_records),
        )
        logger.info("Val: %d, Test: %d", len(val_records), len(test_records))

        ds_p = TwoWayRecordDataset(p_records, self.cfg)
        ds_u = TwoWayRecordDataset(u_records, self.cfg)
        ds_val = TwoWayRecordDataset(val_records, self.cfg)
        ds_test = TwoWayRecordDataset(test_records, self.cfg)

        collate = lambda b: twoway_collate_fn(b, self.cfg)

        loaders = {
            "train_p": DataLoader(
                ds_p, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, collate_fn=collate,
            ),
            "train_u": DataLoader(
                ds_u, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, collate_fn=collate,
            ),
            "val": DataLoader(
                ds_val, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate,
            ),
            "test": DataLoader(
                ds_test, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, collate_fn=collate,
            ),
        }
        return loaders
