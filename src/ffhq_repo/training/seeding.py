"""Global RNG seeding, kept separate from the deterministic hashlib-based
data-split seeding in :mod:`ffhq_repo.data.dataset` (so changing the
training seed never perturbs which images were selected for train/val).
"""
from __future__ import annotations

import random

import numpy as np
import torch

__all__ = ["set_all_seeds"]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
