# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""CosMAP graph construction.
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
    faiss_backend: str = "none", #auto
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
        # if verbose:
        #     print(f"{ts()} Using sklearn NearestNeighbors CPU fallback")
        # Y_cpu = Y.detach().cpu().numpy()
        # nbrs = NearestNeighbors(
        #     n_neighbors=n_neighbors + 1,
        #     algorithm="auto",
        #     metric="cosine",
        #     n_jobs=-1,
        # ).fit(Y_cpu)
        # _, indices = nbrs.kneighbors(Y_cpu)
        # neighbors_indices = torch.as_tensor(indices[:, 1:], dtype=torch.long, device=device)
        # del Y_cpu, indices
        from pynndescent import NNDescent

        if verbose:
            print(f"{ts()} Using PyNNDescent CPU fallback (Cosine)")

        Y_cpu = Y.detach().cpu().numpy().astype("float32", copy=False)

        index = NNDescent(
            Y_cpu,
            n_neighbors=n_neighbors + 1,
            metric="cosine",
            n_jobs=-1,
            random_state=42,
            verbose=verbose,
        )

        indices, _ = index.neighbor_graph

        neighbors_indices = torch.as_tensor(
            indices[:, 1:],
            dtype=torch.long,
            device=device,
        )

        del Y_cpu, indices, index

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


def compute_euclidean_similarity_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    temperature: float,
    use_gpu: bool = True,
    batch_size: int = 1000,
    faiss_backend: str = "auto",
    verbose: bool = False,
) -> sp.csr_matrix:
    """
    Compute CosMAP's sparse temperature-scaled Euclidean similarity graph.
    It finds nearest neighbors under Euclidean distance and converts distances
    into local probabilities using a softmax over negative squared distances:

        p_ij ∝ exp(-d(x_i - x_j)^2 / temperature)/sum_{k in Nk(i)} exp(-d(x_i - x_k)^2 / temperature)

    where Nk(i) is the set of k nearest neighbors of x_i under Euclidean distance.
    """
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
            "The imported FAISS package does not provide StandardGpuResources/GpuIndexFlatL2, "
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
        print(f"{ts()} Preparing input data on {device} (no normalization for Euclidean)")

    Y = X_tensor

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
        print(f"{ts()} Finding k-nearest neighbors (Euclidean distance)")

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
            print(f"{ts()} Using FAISS GPU IndexFlatL2")
        Y_np = Y.detach().cpu().numpy().astype(np.float32)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, Y_np.shape[1])
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
            print(f"{ts()} Using batched torch {device.type.upper()} k-NN (Euclidean)")

        neighbors_indices = torch.zeros((n_samples, n_neighbors), dtype=torch.long, device=device)
        k_nn_batch_size = min(batch_size * 5, 5000)

        for i in tqdm(range(0, n_samples, k_nn_batch_size), desc=f"Finding k-NN on {device.type.upper()}", disable=not verbose):
            end_idx = min(i + k_nn_batch_size, n_samples)
            batch_Y = Y[i:end_idx]
            
            # Compute squared Euclidean distances
            sq_dist = torch.cdist(batch_Y, Y, p=2.0) ** 2

            row_ids = torch.arange(end_idx - i, device=device)
            col_ids = torch.arange(i, end_idx, device=device)
            sq_dist[row_ids, col_ids] = float("inf")

            _, topk_indices = torch.topk(sq_dist, n_neighbors, dim=1, largest=False)
            neighbors_indices[i:end_idx] = topk_indices

            del sq_dist, topk_indices
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    else:
        # if verbose:
        #     print(f"{ts()} Using sklearn NearestNeighbors CPU fallback (Euclidean)")
        # Y_cpu = Y.detach().cpu().numpy()
        # nbrs = NearestNeighbors(
        #     n_neighbors=n_neighbors + 1,
        #     algorithm="auto",
        #     metric="euclidean",
        #     n_jobs=-1,
        # ).fit(Y_cpu)
        # _, indices = nbrs.kneighbors(Y_cpu)
        # neighbors_indices = torch.as_tensor(indices[:, 1:], dtype=torch.long, device=device)
        # del Y_cpu, indices
        from pynndescent import NNDescent

        if verbose:
            print(f"{ts()} Using PyNNDescent CPU fallback (Euclidean)")

        Y_cpu = Y.detach().cpu().numpy().astype("float32", copy=False)

        index = NNDescent(
            Y_cpu,
            n_neighbors=n_neighbors + 1,
            metric="euclidean",
            n_jobs=-1,
            random_state=42,
            verbose=verbose,
        )

        indices, _ = index.neighbor_graph

        neighbors_indices = torch.as_tensor(
            indices[:, 1:],
            dtype=torch.long,
            device=device,
        )

        del Y_cpu, indices, index

    if verbose:
        print(f"{ts()} Building sparse temperature-softmax graph (Euclidean)")

    rows_all = []
    cols_all = []
    vals_all = []

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing similarities", disable=not verbose):
        end_idx = min(i + batch_size, n_samples)

        batch_rows = torch.arange(i, end_idx, device=device, dtype=torch.long)
        neigh = neighbors_indices[i:end_idx]

        y_i = Y[batch_rows]
        y_j = Y[neigh]

        # Compute squared Euclidean distances
        sq_distances = torch.sum((y_i.unsqueeze(1) - y_j) ** 2, dim=2)
        
        # Convert to similarities: exp(-d^2 / temperature)
        similarities = torch.exp(-sq_distances / temperature)
        
        # Apply softmax normalization and scale by 0.5
        probs = torch.softmax(similarities, dim=1) * 0.5

        source_rows = batch_rows.unsqueeze(1).expand(-1, n_neighbors).reshape(-1)
        target_cols = neigh.reshape(-1)
        values = probs.reshape(-1)

        rows = torch.cat([source_rows, target_cols], dim=0).detach().cpu().numpy()
        cols = torch.cat([target_cols, source_rows], dim=0).detach().cpu().numpy()
        vals = torch.cat([values, values], dim=0).detach().cpu().numpy()

        rows_all.append(rows)
        cols_all.append(cols)
        vals_all.append(vals)

        del y_i, y_j, sq_distances, similarities, probs, source_rows, target_cols, values
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


def compute_precomputed_similarity_graph(
    X: np.ndarray,
    *,
    n_neighbors: int,
    temperature: float,
    use_gpu: bool = True,
    batch_size: int = 1000,
    faiss_backend: str = "auto",
    verbose: bool = False,
) -> sp.csr_matrix:
    """
    Compute CosMAP's sparse temperature-scaled graph from a precomputed
    pairwise similarity matrix.

    The input X must be an n_samples x n_samples matrix where X[i, j] is
    already a high-dimensional similarity score between observations i and j.
    For each row i, the k largest off-diagonal similarities define N_k(i), and
    the local edge weights are computed with the same temperature-softmax logic
    used in the paper:

        p_ij = softmax(X[i, N_k(i)] / temperature)_j * 0.5

    The final graph is symmetrized by adding both (i, j) and (j, i), matching
    the construction used by the cosine and Euclidean graph builders.
    """
    # Kept for API compatibility with compute_similarity_graph. They are not
    # needed because nearest-neighbor search is performed directly on X.
    _ = faiss_backend

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    if sp.issparse(X):
        if X.ndim != 2:
            raise ValueError("precomputed similarity input must be a 2D matrix")
        if X.shape[0] != X.shape[1]:
            raise ValueError(
                "precomputed similarity input must be square with shape "
                "(n_samples, n_samples)"
            )
        if X.data.size and not np.all(np.isfinite(X.data)):
            raise ValueError("precomputed similarity input contains NaN or infinite values")
        n_samples = int(X.shape[0])
        X_csr = X.tocsr().astype(np.float32, copy=False)
        X_tensor = None
    else:
        if isinstance(X, torch.Tensor):
            if X.ndim != 2:
                raise ValueError("precomputed similarity input must be a 2D matrix")
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    "precomputed similarity input must be square with shape "
                    "(n_samples, n_samples)"
                )
            if not torch.isfinite(X).all().item():
                raise ValueError("precomputed similarity input contains NaN or infinite values")
            n_samples = int(X.shape[0])
            device = get_torch_device(use_gpu=use_gpu)
            X_tensor = X.float().to(device)
        else:
            X_arr = np.asarray(X)
            if X_arr.ndim != 2:
                raise ValueError("precomputed similarity input must be a 2D matrix")
            if X_arr.shape[0] != X_arr.shape[1]:
                raise ValueError(
                    "precomputed similarity input must be square with shape "
                    "(n_samples, n_samples)"
                )
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("precomputed similarity input contains NaN or infinite values")
            n_samples = int(X_arr.shape[0])
            device = get_torch_device(use_gpu=use_gpu)
            X_tensor = torch.from_numpy(X_arr.astype(np.float32, copy=False)).to(device)

        X_csr = None

    if n_samples < 2:
        raise ValueError("precomputed similarity input must contain at least two samples")
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be smaller than the number of samples")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if verbose:
        print(f"{ts()} Building sparse graph from precomputed similarities")

    rows_all = []
    cols_all = []
    vals_all = []

    if X_csr is not None:
        # Sparse input is accepted for convenience. Each row is densified only
        # for the top-k operation, while the output remains sparse.
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing precomputed graph", disable=not verbose):
            end_idx = min(i + batch_size, n_samples)
            batch = X_csr[i:end_idx].toarray().astype(np.float32, copy=False)
            diag_cols = np.arange(i, end_idx)
            batch[np.arange(end_idx - i), diag_cols] = -np.inf

            topk_unsorted = np.argpartition(-batch, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
            topk_vals_unsorted = np.take_along_axis(batch, topk_unsorted, axis=1)
            order = np.argsort(-topk_vals_unsorted, axis=1)
            neigh = np.take_along_axis(topk_unsorted, order, axis=1)
            sims = np.take_along_axis(topk_vals_unsorted, order, axis=1)

            sims = sims / float(temperature)
            sims = sims - np.max(sims, axis=1, keepdims=True)
            probs = np.exp(sims)
            probs = (probs / np.sum(probs, axis=1, keepdims=True)) * 0.5

            source_rows = np.repeat(np.arange(i, end_idx, dtype=np.int64), n_neighbors)
            target_cols = neigh.reshape(-1).astype(np.int64)
            values = probs.reshape(-1).astype(np.float32)

            rows_all.append(np.concatenate([source_rows, target_cols]))
            cols_all.append(np.concatenate([target_cols, source_rows]))
            vals_all.append(np.concatenate([values, values]))
    else:
        device = X_tensor.device
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing precomputed graph", disable=not verbose):
            end_idx = min(i + batch_size, n_samples)
            batch = X_tensor[i:end_idx].clone()

            row_ids = torch.arange(end_idx - i, device=device)
            col_ids = torch.arange(i, end_idx, device=device)
            batch[row_ids, col_ids] = -float("inf")

            sims, neigh = torch.topk(batch, n_neighbors, dim=1, largest=True)
            probs = torch.softmax(sims / temperature, dim=1) * 0.5

            source_rows = torch.arange(i, end_idx, device=device, dtype=torch.long).unsqueeze(1).expand(-1, n_neighbors).reshape(-1)
            target_cols = neigh.reshape(-1)
            values = probs.reshape(-1)

            rows = torch.cat([source_rows, target_cols], dim=0).detach().cpu().numpy()
            cols = torch.cat([target_cols, source_rows], dim=0).detach().cpu().numpy()
            vals = torch.cat([values, values], dim=0).detach().cpu().numpy()

            rows_all.append(rows)
            cols_all.append(cols)
            vals_all.append(vals)

            del batch, sims, neigh, probs, source_rows, target_cols, values
            if device.type == "cuda" and (i // max(batch_size, 1)) % 5 == 0:
                torch.cuda.empty_cache()
            elif device.type == "mps" and (i // max(batch_size, 1)) % 5 == 0:
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        del X_tensor
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

def compute_similarity_graph(
    X: np.ndarray,
    *,
    metric: str = "cosine",
    n_neighbors: int,
    temperature: float,
    use_gpu: bool = True,
    batch_size: int = 1000,
    faiss_backend: str = "auto",
    verbose: bool = False,
) -> sp.csr_matrix:
    """
    Compute a sparse CosMAP graph using either cosine or Euclidean , precomputed geometry.

    Parameters
    ----------
    metric : {"cosine", "euclidean","precomputed"}, default="cosine"
        Distance/similarity geometry used to construct the high-dimensional graph.
        - "cosine": L2-normalizes X and uses inner products.
        - "euclidean": uses raw X and converts squared Euclidean distances into similarities.
        -"precomputed": uses raw X as distance already computed
    """
    metric = str(metric).lower()

    if metric == "cosine":
        return compute_cosine_similarity_graph(
            X,
            n_neighbors=n_neighbors,
            temperature=temperature,
            use_gpu=use_gpu,
            batch_size=batch_size,
            faiss_backend=faiss_backend,
            verbose=verbose,
        )

    if metric == "euclidean":
        return compute_euclidean_similarity_graph(
            X,
            n_neighbors=n_neighbors,
            temperature=temperature,
            use_gpu=use_gpu,
            batch_size=batch_size,
            faiss_backend=faiss_backend,
            verbose=verbose,
        )

    if metric == "precomputed":
        return compute_precomputed_similarity_graph(
            X,
            n_neighbors=n_neighbors,
            temperature=temperature,
            use_gpu=use_gpu,
            batch_size=batch_size,
            faiss_backend=faiss_backend,
            verbose=verbose,
        )

    raise ValueError("metric must be one of: 'cosine', 'euclidean', or 'precomputed'")


