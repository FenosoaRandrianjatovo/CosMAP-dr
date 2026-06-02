# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""Embedding initialization helpers for CosMAP."""
from __future__ import annotations

from typing import Optional, Union
from warnings import warn

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KDTree

try:
    from umap.spectral import spectral_layout, tswspectral_layout
    _HAS_UMAP = True
except Exception:
    spectral_layout = None
    tswspectral_layout = None
    _HAS_UMAP = False


def noisy_scale_coords(
    coords: np.ndarray,
    random_state: np.random.RandomState,
    max_coord: float = 10.0,
    noise: float = 0.0001,
) -> np.ndarray:
    """Scale coordinates and add tiny noise to break exact duplicates."""
    max_abs = np.abs(coords).max()
    if max_abs == 0:
        max_abs = 1.0
    expansion = max_coord / max_abs
    coords = (coords * expansion).astype(np.float32)
    return coords + random_state.normal(scale=noise, size=coords.shape).astype(np.float32)


def normalize_embedding_to_10(embedding: np.ndarray) -> np.ndarray:
    """Normalize embedding coordinates into approximately [0, 10]."""
    emb = embedding.astype(np.float32, copy=True)
    min_v = np.min(emb, axis=0)
    max_v = np.max(emb, axis=0)
    denom = max_v - min_v
    denom[denom == 0] = 1.0
    emb = 10.0 * (emb - min_v) / denom
    return np.asarray(emb, dtype=np.float32, order="C")


def initialize_embedding(
    data: np.ndarray,
    graph,
    n_components: int,
    init: Union[str, np.ndarray],
    random_state: np.random.RandomState,
    metric: str,
    metric_kwds: Optional[dict],
    seed: Optional[int] = None,
) -> np.ndarray:
    """Initialize embedding using the original CosMAP paper."""
    if isinstance(init, str) and init == "random":
        embedding = random_state.uniform(
            low=-10.0, high=10.0, size=(graph.shape[0], n_components)
        ).astype(np.float32)

    elif isinstance(init, str) and init == "pca":
        if sp.issparse(data):
            pca = TruncatedSVD(n_components=n_components, random_state=random_state)
        else:
            pca = PCA(n_components=n_components, random_state=random_state)
        embedding = pca.fit_transform(data).astype(np.float32)
        embedding = noisy_scale_coords(embedding, random_state, max_coord=10, noise=0.0001)

    elif isinstance(init, str) and init == "spectral":
        if not _HAS_UMAP or spectral_layout is None:
            warn("UMAP spectral_layout not available. Falling back to random initialization.")
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        else:
            embedding = spectral_layout(
                data,
                graph,
                n_components,
                random_state=seed if seed is not None else random_state,
                metric=metric,
                metric_kwds=metric_kwds or {},
            )
            embedding = noisy_scale_coords(embedding, random_state, max_coord=10, noise=0.0001)

    elif isinstance(init, str) and init == "tswspectral":
        if not _HAS_UMAP or tswspectral_layout is None:
            warn("UMAP tswspectral_layout not available. Falling back to random initialization.")
            embedding = random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], n_components)
            ).astype(np.float32)
        else:
            embedding = tswspectral_layout(
                data,
                graph,
                n_components,
                random_state=seed if seed is not None else random_state,
                metric=metric,
                metric_kwds=metric_kwds or {},
            )
            embedding = noisy_scale_coords(embedding, random_state, max_coord=10, noise=0.0001)

    else:
        init_data = np.array(init, dtype=np.float32)
        if len(init_data.shape) != 2:
            raise ValueError("init must be 'random', 'pca', 'spectral', 'tswspectral', or a 2D array")
        if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
            tree = KDTree(init_data)
            dist, _ = tree.query(init_data, k=2)
            nndist = np.mean(dist[:, 1])
            embedding = init_data + random_state.normal(
                scale=0.001 * nndist, size=init_data.shape
            ).astype(np.float32)
        else:
            embedding = init_data.astype(np.float32)

    return normalize_embedding_to_10(embedding)
