from typing import Dict, Tuple

import torch

TensorPair = Tuple[torch.Tensor, torch.Tensor]
TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
Metrics = Dict[str, float]
