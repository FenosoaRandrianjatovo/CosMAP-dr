# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""
The numba_cpu optimizer is a JIT-compiled, multi-threaded CPU implementation
especially for high-dimensional data
"""
import numpy as np
from numba import njit, prange, get_num_threads
from typing import Optional, Tuple, Union


@njit(fastmath=True, cache=True, inline="always")
def _clip(v, c):
    if v > c:
        return c
    if v < -c:
        return -c
    return v


@njit(parallel=True, fastmath=True, cache=True)
def _epoch(
    emb, head, tail, weight,
    n_vertices, dim, a, b, gamma,
    neg_rate, alpha, clip_value,
    rng_states, collect_loss,
):
    """
    One epoch. Faithful to the original:
      - attractive update moves emb[h] += a*g, emb[t] -= a*g
      - repulsive update moves only emb[neg_h] += a*g
      - same gradient coefficients, same clip, same eps
    Returns (sum_pos_loss, sum_neg_loss, n_batches_counted) for the proxy.
    Parallelized over edges with prange; benign races on shared vertex rows
    (acceptable for SGD, same as the GPU index_add_ which is also unordered).
    """
    n_edges = head.shape[0]
    eps = 1e-6
    pos_loss = 0.0
    neg_loss = 0.0

    for i in prange(n_edges):
        j = head[i]
        k = tail[i]
        w = weight[i]

        dist_sq = 0.0
        for d in range(dim):
            diff = emb[j, d] - emb[k, d]
            dist_sq += diff * diff
        if dist_sq < eps:
            dist_sq = eps

        pow_b = dist_sq ** b
        denom = 1.0 + a * pow_b

        if collect_loss:
            q_pos = 1.0 / denom
            if q_pos < 1e-12:
                q_pos = 1e-12
            pos_loss += -w * np.log(q_pos)

        grad_coeff = (-2.0 * a * b * (dist_sq ** (b - 1.0))) / denom

        for d in range(dim):
            diff = emb[j, d] - emb[k, d]
            g = _clip(grad_coeff * diff, clip_value) * w
            emb[j, d] += alpha * g
            emb[k, d] -= alpha * g

        if neg_rate > 0:
            tid = i % rng_states.shape[0]
            for _ in range(neg_rate):
                # xorshift64 from per-thread state
                s = rng_states[tid]
                s ^= s << np.uint64(13)
                s ^= s >> np.uint64(7)
                s ^= s << np.uint64(17)
                rng_states[tid] = s
                kk = np.int64(s % np.uint64(n_vertices))
                if kk == j:
                    kk = (kk + 1) % n_vertices

                ndist_sq = 0.0
                for d in range(dim):
                    diff = emb[j, d] - emb[kk, d]
                    ndist_sq += diff * diff
                if ndist_sq < eps:
                    ndist_sq = eps

                npow_b = ndist_sq ** b

                if collect_loss:
                    q_neg = 1.0 / (1.0 + a * npow_b)
                    one_minus = 1.0 - q_neg
                    if one_minus < 1e-12:
                        one_minus = 1e-12
                    neg_loss += -gamma * np.log(one_minus)

                grad_coeff_rep = (2.0 * gamma * b) / (
                    (0.001 + ndist_sq) * (1.0 + a * npow_b)
                )
                for d in range(dim):
                    diff = emb[j, d] - emb[kk, d]
                    g = _clip(grad_coeff_rep * diff, clip_value)
                    emb[j, d] += alpha * g

    return pos_loss, neg_loss


def optimize_layout_euclidean_numba_cpu(
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
    seed: Optional[int] = None,
    verbose: bool = False,
    clip_value: float = 4.0,
    collect_loss: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, list]]:
    """JIT-compiled, multi-threaded CPU port of the manual optimizer (no torch)."""

    emb = np.array(embedding, dtype=np.float32, copy=True)
    head_t = np.ascontiguousarray(head, dtype=np.int64)
    tail_t = np.ascontiguousarray(tail, dtype=np.int64)
    weight_t = np.ascontiguousarray(weight, dtype=np.float32)

    max_weight = max(float(weight_t.max()), 1e-12)
    weight_t = weight_t / max_weight

    n_edges = int(head_t.shape[0])
    dim = emb.shape[1]
    loss_history = []

    if verbose:
        print(f"numba_cpu optimization | n_vertices={n_vertices}, "
              f"n_edges={n_edges}, n_epochs={n_epochs}, threads={get_num_threads()}")

    # Per-thread RNG states (nonzero uint64), seeded reproducibly.
    n_threads = get_num_threads()
    base = np.random.RandomState(seed if seed is not None else None)
    rng_states = base.randint(1, 2**63 - 1, size=n_threads).astype(np.uint64)

    for epoch in range(int(n_epochs)):
        alpha = initial_alpha * (1.0 - float(epoch) / float(n_epochs))
        alpha = max(alpha, initial_alpha * 0.001)

        pos_loss, neg_loss = _epoch(
            emb, head_t, tail_t, weight_t,
            int(n_vertices), dim, float(a), float(b), float(gamma),
            int(negative_sample_rate), float(alpha), float(clip_value),
            rng_states, bool(collect_loss),
        )

        if collect_loss:
            loss_history.append((pos_loss + neg_loss) / max(n_edges, 1))

        if verbose and (
            epoch == 0
            or epoch == int(n_epochs) - 1
            or (epoch + 1) % max(1, int(n_epochs) // 10) == 0
        ):
            print(f"Epoch {epoch + 1:4d}/{int(n_epochs)} | alpha={alpha:.6f} | device=cpu")

    emb = emb.astype(np.float32)
    if collect_loss:
        return emb, loss_history
    return emb
