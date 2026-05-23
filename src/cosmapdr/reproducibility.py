# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""Reproducibility helpers for CosMAP.

Important design rule
---------------------
CosMAP must not silently replace ``random_state=None`` by a fixed seed.  When
``random_state`` is None, the estimator uses fresh randomness and avoids forcing
PyTorch deterministic algorithms.  When an integer seed is supplied, the random
number generators are seeded.  Full deterministic CUDA behavior is optional
because it can reduce speed.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.utils import check_random_state


RandomStateLike = Optional[Union[int, np.random.RandomState]]


def make_random_state(random_state: RandomStateLike) -> Tuple[np.random.RandomState, Optional[int]]:
    """Return a NumPy RandomState and the original integer seed, if supplied.

    ``sklearn.utils.check_random_state(None)`` returns NumPy's global RandomState.
    That is not fixed, but after that conversion we lose whether the user gave
    None or an explicit seed.  This helper preserves that information.
    """
    if random_state is None:
        return np.random.RandomState(), None
    if isinstance(random_state, (int, np.integer)):
        return check_random_state(int(random_state)), int(random_state)
    if isinstance(random_state, np.random.RandomState):
        return random_state, None
    return check_random_state(random_state), None


def seed_everything(seed: Optional[int], deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch only when ``seed`` is not None."""
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        # Faster default.  This keeps seeded RNG streams but avoids forcing
        # slower deterministic kernels everywhere.
        cudnn.benchmark = True
        cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def child_seed(base_seed: Optional[int], offset: int = 0) -> Optional[int]:
    """Create a deterministic child seed, or None if the parent is unseeded."""
    if base_seed is None:
        return None
    return int((int(base_seed) + int(offset)) % (2**31 - 1))
