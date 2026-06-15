# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""Low-dimensional CosMAP optimization.

"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch

try:
    from umap.layouts import optimize_layout_euclidean
    _HAS_UMAP = True
except Exception:
    optimize_layout_euclidean = None
    _HAS_UMAP = False

from .initialization import initialize_embedding, normalize_embedding_to_10
from .reproducibility import seed_everything
from .utils import INT32_MAX, INT32_MIN, get_torch_device, ts


def make_epochs_per_sample(weights: np.ndarray, n_epochs: int) -> np.ndarray:
    """UMAP helper kept for compatibility with the numba fallback backend."""
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result


@torch.no_grad()
def optimize_layout_euclidean_torch_manual(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    weight: np.ndarray,
    n_epochs: int,
    n_vertices: int,
    a: float,
    b: float,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    initial_alpha: float = 1.0,
    batch_size: int = 65536,
    random_state: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    verbose: bool = False,
    clip_value: float = 4.0,
    collect_loss: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Fast PyTorch optimizer using manual coordinate updates."""
    if device is None:
        device = get_torch_device(use_gpu=use_gpu)

    # Seed only when the user supplied a seed.
    seed_everything(seed, deterministic=deterministic)

    emb = torch.as_tensor(embedding, dtype=torch.float32, device=device).clone()

    head_t = torch.as_tensor(head, dtype=torch.long, device=device)
    tail_t = torch.as_tensor(tail, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)

    max_weight = torch.max(weight_t).clamp_min(1e-12)
    weight_t = weight_t / max_weight

    n_edges = int(head_t.shape[0])
    eps = 1e-6
    loss_history = []

    if verbose:
        print(f"{ts()} Starting torch_manual optimization on {device}")
        print(f"{ts()} n_vertices={n_vertices}, n_edges={n_edges}, n_epochs={n_epochs}")

    # Local generators 
    generator = None
    if device.type in {"cuda", "cpu"} and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    for epoch in range(int(n_epochs)):
        alpha = initial_alpha * (1.0 - float(epoch) / float(n_epochs))
        alpha = max(alpha, initial_alpha * 0.001)

        epoch_loss = 0.0
        n_loss_batches = 0
        if generator is None:
            perm = torch.randperm(n_edges, device=device)
        else:
            perm = torch.randperm(n_edges, device=device, generator=generator)

        for start in range(0, n_edges, batch_size):
            idx = perm[start : start + batch_size]

            h = head_t[idx]
            t = tail_t[idx]
            w = weight_t[idx].unsqueeze(1)

            y_h = emb[h]
            y_t = emb[t]

            diff = y_h - y_t
            dist_sq = torch.sum(diff * diff, dim=1, keepdim=True).clamp_min(eps)

            if collect_loss:
                q_pos = 1.0 / (1.0 + a * torch.pow(dist_sq, b))
                pos_loss_proxy = -(w * torch.log(q_pos.clamp_min(1e-12))).mean()
            else:
                pos_loss_proxy = None

            grad_coeff_attr = (
                -2.0
                * a
                * b
                * torch.pow(dist_sq, b - 1.0)
                / (1.0 + a * torch.pow(dist_sq, b))
            )
            grad_attr = grad_coeff_attr * diff
            grad_attr = torch.clamp(grad_attr, -clip_value, clip_value)
            grad_attr = w * grad_attr

            emb.index_add_(0, h, alpha * grad_attr)
            emb.index_add_(0, t, -alpha * grad_attr)

            if negative_sample_rate > 0:
                neg_h = h.repeat_interleave(int(negative_sample_rate))
                if generator is None:
                    neg_t = torch.randint(
                        low=0,
                        high=int(n_vertices),
                        size=(neg_h.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    neg_t = torch.randint(
                        low=0,
                        high=int(n_vertices),
                        size=(neg_h.shape[0],),
                        device=device,
                        dtype=torch.long,
                        generator=generator,
                    )

                same = neg_h == neg_t
                if same.any():
                    neg_t[same] = (neg_t[same] + 1) % int(n_vertices)

                y_nh = emb[neg_h]
                y_nt = emb[neg_t]

                neg_diff = y_nh - y_nt
                neg_dist_sq = torch.sum(neg_diff * neg_diff, dim=1, keepdim=True).clamp_min(eps)

                if collect_loss:
                    q_neg = 1.0 / (1.0 + a * torch.pow(neg_dist_sq, b))
                    neg_loss_proxy = -gamma * torch.log((1.0 - q_neg).clamp_min(1e-12)).mean()
                    epoch_loss += float((pos_loss_proxy + neg_loss_proxy).detach().cpu())
                    n_loss_batches += 1

                grad_coeff_rep = (
                    2.0
                    * gamma
                    * b
                    / ((0.001 + neg_dist_sq) * (1.0 + a * torch.pow(neg_dist_sq, b)))
                )
                grad_rep = grad_coeff_rep * neg_diff
                grad_rep = torch.clamp(grad_rep, -clip_value, clip_value)

                emb.index_add_(0, neg_h, alpha * grad_rep)

            elif collect_loss and pos_loss_proxy is not None:
                epoch_loss += float(pos_loss_proxy.detach().cpu())
                n_loss_batches += 1

        if collect_loss:
            loss_history.append(epoch_loss / max(n_loss_batches, 1))

        if verbose and (
            epoch == 0
            or epoch == int(n_epochs) - 1
            or (epoch + 1) % max(1, int(n_epochs) // 10) == 0
        ):
            print(f"{ts()} Epoch {epoch + 1:4d}/{int(n_epochs)} | alpha={alpha:.6f} | device={device}")

    embedding_np = emb.detach().cpu().numpy().astype(np.float32)
    if collect_loss:
        return embedding_np, loss_history
    return embedding_np


def optimize_layout_euclidean_torch_autograd(
    embedding: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    weight: np.ndarray,
    n_epochs: int,
    n_vertices: int,
    a: float,
    b: float,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    initial_alpha: float = 1.0,
    batch_size: int = 65536,
    random_state: Optional[np.random.RandomState] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    verbose: bool = False,
    collect_loss: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """Autograd-based PyTorch optimizer."""
    if device is None:
        device = get_torch_device(use_gpu=use_gpu)

    seed_everything(seed, deterministic=deterministic)

    emb = torch.nn.Parameter(torch.as_tensor(embedding, dtype=torch.float32, device=device).clone())

    head_t = torch.as_tensor(head, dtype=torch.long, device=device)
    tail_t = torch.as_tensor(tail, dtype=torch.long, device=device)
    weight_t = torch.as_tensor(weight, dtype=torch.float32, device=device)
    weight_t = weight_t / torch.max(weight_t).clamp_min(1e-12)

    n_edges = int(head_t.shape[0])
    optimizer = torch.optim.Adam([emb], lr=initial_alpha)
    eps = 1e-6
    loss_history = []

    generator = None
    if device.type in {"cuda", "cpu"} and seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    if verbose:
        print(f"{ts()} Starting torch_autograd optimization on {device}")
        print(f"{ts()} n_vertices={n_vertices}, n_edges={n_edges}, n_epochs={n_epochs}")

    for epoch in range(int(n_epochs)):
        lr = initial_alpha * (1.0 - float(epoch) / float(n_epochs))
        lr = max(lr, initial_alpha * 0.001)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if generator is None:
            perm = torch.randperm(n_edges, device=device)
        else:
            perm = torch.randperm(n_edges, device=device, generator=generator)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_edges, batch_size):
            idx = perm[start : start + batch_size]
            h = head_t[idx]
            t = tail_t[idx]
            w = weight_t[idx]

            y_h = emb[h]
            y_t = emb[t]
            dist_sq = torch.sum((y_h - y_t) ** 2, dim=1).clamp_min(eps)
            q_pos = 1.0 / (1.0 + a * torch.pow(dist_sq, b))
            pos_loss = -w * torch.log(q_pos + 1e-12)

            if negative_sample_rate > 0:
                neg_h = h.repeat_interleave(int(negative_sample_rate))
                if generator is None:
                    neg_t = torch.randint(low=0, high=int(n_vertices), size=(neg_h.shape[0],), device=device, dtype=torch.long)
                else:
                    neg_t = torch.randint(low=0, high=int(n_vertices), size=(neg_h.shape[0],), device=device, dtype=torch.long, generator=generator)
                same = neg_h == neg_t
                if same.any():
                    neg_t[same] = (neg_t[same] + 1) % int(n_vertices)

                y_nh = emb[neg_h]
                y_nt = emb[neg_t]
                neg_dist_sq = torch.sum((y_nh - y_nt) ** 2, dim=1).clamp_min(eps)
                q_neg = 1.0 / (1.0 + a * torch.pow(neg_dist_sq, b))
                neg_loss = -gamma * torch.log(1.0 - q_neg + 1e-12)
                loss = pos_loss.mean() + neg_loss.mean()
            else:
                loss = pos_loss.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

        if collect_loss:
            loss_history.append(epoch_loss / max(n_batches, 1))

        if verbose and (
            epoch == 0
            or epoch == int(n_epochs) - 1
            or (epoch + 1) % max(1, int(n_epochs) // 10) == 0
        ):
            print(f"{ts()} Epoch {epoch + 1:4d}/{int(n_epochs)} | loss={epoch_loss / max(n_batches, 1):.6f} | lr={lr:.6f} | device={device}")

    embedding_np = emb.detach().cpu().numpy().astype(np.float32)
    if collect_loss:
        return embedding_np, loss_history
    return embedding_np


def optimization_simplicial_set(
    data: np.ndarray,
    graph: sp.spmatrix,
    n_components: int,
    initial_alpha: float,
    a: float,
    b: float,
    gamma: float,
    negative_sample_rate: int,
    n_epochs: Optional[Union[int, list]],
    init: Union[str, np.ndarray],
    random_state: np.random.RandomState,
    seed: Optional[int],
    deterministic: bool,
    metric: str,
    metric_kwds: Optional[dict],
    densmap: bool,
    densmap_kwds: dict,
    output_dens: bool,
    output_metric: Optional[Any] = None,
    output_metric_kwds: dict = {},
    euclidean_output: bool = True,
    parallel: bool = False,
    verbose: bool = False,
    tqdm_kwds: Optional[dict] = None,
    optimizer_backend: str = "torch_manual",
    use_gpu: bool = True,
    torch_batch_size: int = 65536,
    collect_loss: bool = False,
):
    """Perform CosMAP/UMAP-style fuzzy simplicial set embedding."""
    if not euclidean_output:
        raise NotImplementedError("CosMAP torch optimization supports euclidean_output=True only.")

    if densmap or output_dens:
        raise NotImplementedError(
            "CosMAP torch optimizer does not implement densMAP/output_dens. "
            "Use optimizer_backend='umap' if you need the original UMAP density extensions."
        )

    graph = graph.tocoo()
    graph.sum_duplicates()
    n_vertices = graph.shape[1]

    default_epochs = 500 if graph.shape[0] <= 10000 else 200
    if n_epochs is None:
        n_epochs = default_epochs
    n_epochs_max = max(n_epochs) if isinstance(n_epochs, list) else int(n_epochs)

    if n_epochs_max > 10:
        graph.data[graph.data < (graph.data.max() / float(n_epochs_max))] = 0.0
    else:
        graph.data[graph.data < (graph.data.max() / float(default_epochs))] = 0.0

    graph.eliminate_zeros()
    graph = graph.tocoo()
    graph.sum_duplicates()

    seed_everything(seed, deterministic=deterministic)

    embedding = initialize_embedding(
        data=data,
        graph=graph,
        n_components=n_components,
        init=init,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds or {},
        seed=seed,
    )
    embedding = normalize_embedding_to_10(embedding)

    head = graph.row.astype(np.int64)
    tail = graph.col.astype(np.int64)
    weight = graph.data.astype(np.float32)

    aux_data = {}
    optimizer_backend = optimizer_backend.lower()

    if optimizer_backend == "torch_manual":
        optimizer_result = optimize_layout_euclidean_torch_manual(
            embedding=embedding,
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs_max,
            n_vertices=n_vertices,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            initial_alpha=initial_alpha,
            batch_size=torch_batch_size,
            random_state=random_state,
            seed=seed,
            deterministic=deterministic,
            device=None,
            use_gpu=use_gpu,
            verbose=verbose,
            collect_loss=collect_loss,
        )
        if isinstance(optimizer_result, tuple):
            embedding, loss_history = optimizer_result
            aux_data["loss_history"] = loss_history
        else:
            embedding = optimizer_result

            
    elif optimizer_backend == "cpu_numba_manual":
        from .numba_optimization import optimize_layout_euclidean_numba_cpu
        
        optimizer_result = optimize_layout_euclidean_numba_cpu(
            embedding=embedding,
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs_max,
            n_vertices=n_vertices,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            initial_alpha=initial_alpha,
            seed=seed,
            verbose=verbose,
            collect_loss=collect_loss,
        )
        if isinstance(optimizer_result, tuple):
            embedding, loss_history = optimizer_result
            aux_data["loss_history"] = loss_history
        else:
            embedding = optimizer_result


    elif optimizer_backend == "torch_autograd":
        optimizer_result = optimize_layout_euclidean_torch_autograd(
            embedding=embedding,
            head=head,
            tail=tail,
            weight=weight,
            n_epochs=n_epochs_max,
            n_vertices=n_vertices,
            a=a,
            b=b,
            gamma=gamma,
            negative_sample_rate=negative_sample_rate,
            initial_alpha=initial_alpha,
            batch_size=torch_batch_size,
            random_state=random_state,
            seed=seed,
            deterministic=deterministic,
            device=None,
            use_gpu=use_gpu,
            verbose=verbose,
            collect_loss=collect_loss,
        )
        if isinstance(optimizer_result, tuple):
            embedding, loss_history = optimizer_result
            aux_data["loss_history"] = loss_history
        else:
            embedding = optimizer_result

    elif optimizer_backend == "umap":
        if not _HAS_UMAP or optimize_layout_euclidean is None:
            raise ImportError("umap-learn is required for optimizer_backend='umap'.")
        epochs_per_sample = make_epochs_per_sample(weight, n_epochs_max)
        rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        embedding = optimize_layout_euclidean(
            embedding,
            embedding,
            head,
            tail,
            n_epochs,
            n_vertices,
            epochs_per_sample,
            a,
            b,
            rng_state,
            gamma,
            initial_alpha,
            negative_sample_rate,
            parallel=parallel,
            verbose=verbose,
            densmap=densmap,
            densmap_kwds=densmap_kwds,
            tqdm_kwds=tqdm_kwds,
            move_other=True,
        )
        if isinstance(embedding, list):
            aux_data["embedding_list"] = embedding
            embedding = embedding[-1].copy()

    else:
        raise ValueError("optimizer_backend must be one of: 'torch_manual', 'torch_autograd', or 'umap'.")

    return embedding.astype(np.float32), aux_data
