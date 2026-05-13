# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""CosMAP graph construction.

1. L2-normalize high-dimensional data.
2. Find k-nearest neighbors using inner product over normalized vectors.
3. Convert neighbor cosine scores to temperature-scaled softmax weights.
4. Symmetrize the sparse graph by adding both i->j and j->i.
"""
from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .metrics import l2_normalize
from .utils import get_torch_device, safe_import_faiss, ts


def compute_cosine_similarity_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    temperature: float,
    use_gpu: bool = True,
    batch_size: int = 1000,
    faiss_backend: str = "auto",
    verbose: bool = False,
) -> sp.csr_matrix:
    """Compute CosMAP's sparse temperature-scaled cosine similarity graph."""
    faiss_backend = str(faiss_backend).lower()
    if faiss_backend not in {"auto", "gpu", "none"}:
        raise ValueError("faiss_backend must be one of: 'auto', 'gpu', or 'none'. FAISS-CPU fallback is intentionally disabled.")

    if faiss_backend == "none":
        faiss = None
        faiss_has_cpu = False
        faiss_has_gpu = False
        if verbose:
            print(f"{ts()} FAISS disabled by faiss_backend='none'")
    else:
        faiss, faiss_has_cpu, faiss_has_gpu = safe_import_faiss(verbose=verbose)

    if faiss_backend == "gpu" and (not torch.cuda.is_available() or not faiss_has_gpu):
        raise RuntimeError(
            "faiss_backend='gpu' was requested, but CUDA and FAISS-GPU are not both available. "
            "The imported FAISS package does not provide StandardGpuResources/GpuIndexFlatIP, "
            "or PyTorch CUDA is unavailable. Fix the environment/module so FAISS-GPU is imported, "
            "or use faiss_backend='auto' to allow torch GPU/sklearn CPU fallback."
        )

    if faiss is not None and faiss_has_cpu and not faiss_has_gpu and verbose:
        print(
            f"{ts()} FAISS-CPU was detected but will not be used automatically. "
            "CosMAP uses FAISS only when CUDA + FAISS-GPU are both available."
        )

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = get_torch_device(use_gpu=use_gpu)

    if isinstance(X, np.ndarray):
        X_tensor = torch.from_numpy(X).float()
    else:
        X_tensor = X.float()

    X_tensor = X_tensor.to(device)
    n_samples = int(X_tensor.shape[0])

    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    if verbose:
        print(f"{ts()} Normalizing input data on {device}")

    Y = l2_normalize(X_tensor)

    del X_tensor
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    if verbose:
        print(f"{ts()} Finding k-nearest neighbors")

    use_faiss_gpu_now = (
        faiss is not None
        and faiss_has_gpu
        and faiss_backend in {"auto", "gpu"}
        and use_gpu
        and torch.cuda.is_available()
        and device.type == "cuda"
    )

    if use_faiss_gpu_now:
        if verbose:
            print(f"{ts()} Using FAISS GPU IndexFlatIP")
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, Y_np.shape[1])
        index.add(Y_np)
        _, I = index.search(Y_np, n_neighbors + 1)
        neighbors_indices = torch.as_tensor(I[:, 1:], dtype=torch.long, device=device)
        del Y_np, I, index, res
        gc.collect()

    elif use_gpu and device.type in {"cuda", "mps"}:
        if verbose and device.type == "cuda" and faiss is not None and not faiss_has_gpu:
            print(
                f"{ts()} CUDA is available, but the imported FAISS has no GPU symbols; "
                "skipping FAISS-CPU and using batched torch GPU k-NN instead."
            )
        if verbose and device.type == "mps":
            print(
                f"{ts()} MPS is available on this Mac; "
                "FAISS-GPU is CUDA-only here, so using batched torch MPS k-NN instead."
            )
        if verbose:
            print(f"{ts()} Using batched torch {device.type.upper()} k-NN")

        neighbors_indices = torch.zeros((n_samples, n_neighbors), dtype=torch.long, device=device)
        k_nn_batch_size = min(batch_size * 5, 5000)

        for i in tqdm(range(0, n_samples, k_nn_batch_size), desc=f"Finding k-NN on {device.type.upper()}", disable=not verbose):
            end_idx = min(i + k_nn_batch_size, n_samples)
            batch_Y = Y[i:end_idx]
            similarity = torch.mm(batch_Y, Y.t())

            row_ids = torch.arange(end_idx - i, device=device)
            col_ids = torch.arange(i, end_idx, device=device)
            similarity[row_ids, col_ids] = -float("inf")

            _, topk_indices = torch.topk(similarity, n_neighbors, dim=1)
            neighbors_indices[i:end_idx] = topk_indices

            del similarity, topk_indices
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    else:
        if verbose:
            print(f"{ts()} Using sklearn NearestNeighbors CPU fallback")
        Y_cpu = Y.detach().cpu().numpy()
        nbrs = NearestNeighbors(
            n_neighbors=n_neighbors + 1,
            algorithm="auto",
            metric="cosine",
        ).fit(Y_cpu)
        _, indices = nbrs.kneighbors(Y_cpu)
        neighbors_indices = torch.as_tensor(indices[:, 1:], dtype=torch.long, device=device)
        del Y_cpu, indices

    if verbose:
        print(f"{ts()} Building sparse temperature-softmax graph")

    rows_all = []
    cols_all = []
    vals_all = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing similarities", disable=not verbose):
        end_idx = min(i + batch_size, n_samples)

        batch_rows = torch.arange(i, end_idx, device=device, dtype=torch.long)
        neigh = neighbors_indices[i:end_idx]

        y_i = Y[batch_rows]
        y_j = Y[neigh]

        dots = torch.sum(y_i.unsqueeze(1) * y_j, dim=2)
        probs = torch.softmax(dots / temperature, dim=1) * 0.5

        source_rows = batch_rows.unsqueeze(1).expand(-1, n_neighbors).reshape(-1)
        target_cols = neigh.reshape(-1)
        values = probs.reshape(-1)

        rows = torch.cat([source_rows, target_cols], dim=0).detach().cpu().numpy()
        cols = torch.cat([target_cols, source_rows], dim=0).detach().cpu().numpy()
        vals = torch.cat([values, values], dim=0).detach().cpu().numpy()

        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)

        del y_i, y_j, dots, probs, source_rows, target_cols, values
        if device.type == "cuda" and (i // max(batch_size, 1)) % 5 == 0:
            torch.cuda.empty_cache()
        elif device.type == "mps" and (i // max(batch_size, 1)) % 5 == 0:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    del Y, neighbors_indices
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    rows = np.concatenate(rows_all).astype(np.int64)
    cols = np.concatenate(cols_all).astype(np.int64)
    vals = np.concatenate(vals_all).astype(np.float32)

    sparse_matrix = sp.coo_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    sparse_matrix.sum_duplicates()
    return sparse_matrix.tocsr()
