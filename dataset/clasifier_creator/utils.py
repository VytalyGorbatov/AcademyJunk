import logging
import random
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Sets seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Global seed set to %s", seed)


class DataUtils:
    """Static utility methods for data processing."""

    @staticmethod
    def buffer_to_fixed_array(buf: str, fixed_len: int) -> np.ndarray:
        """Converts a string buffer to a normalized float array of fixed length."""
        codes = [ord(c) for c in buf]
        arr = np.zeros(fixed_len, dtype=np.float32)
        n = min(len(codes), fixed_len)
        if n > 0:
            arr[:n] = codes[:n]
        arr /= 255.0
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
