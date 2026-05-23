# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""Similarity metrics used by CosMAP.

CosMAP's default graph is based on cosine similarity.  This file is separated
so new similarity functions can be added without touching the estimator class or
layout optimizer.
"""
from __future__ import annotations

import torch


def l2_normalize(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize rows of a tensor."""
    norms = torch.norm(X, p=2, dim=1, keepdim=True).clamp_min(eps)
    return X / norms


def cosine_pairwise_rows(x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between aligned rows of already-normalized tensors."""
    return torch.sum(x_i * x_j, dim=-1)
