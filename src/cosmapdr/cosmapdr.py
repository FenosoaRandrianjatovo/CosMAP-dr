# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>

"""
CosMAP: Constrastive  Manifold Approximation and Projection

A dimensionality reduction technique that leverages Normalized cross Entropy Similarities
and manifold learning principles for embedding high-dimensional data.
"""
from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .kernels import find_ab_
from .optimization import optimization_simplicial_set
from .reproducibility import child_seed, make_random_state
from .similarities import compute_similarity_graph
from .utils import cleanup_torch_memory, get_torch_device, ts


class CosMAP(BaseEstimator, TransformerMixin):
    """CosMAP: Contrastive Manifold Approximation and Projection."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        temperature: float = 0.5,
        n_epochs: Optional[int] = None,
        learning_rate: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        random_state: Optional[int] = None,
        deterministic: bool = False,
        verbose: bool = False,
        use_gpu: bool = True,
        batch_size: int = 1000,
        metric: str = "cosine",  #euclidean but precomputed  is not yet fully optimized
        init: Union[str, np.ndarray] = "spectral",
        optimizer_backend: str = "torch_manual",
        torch_batch_size: int = 65536,
        negative_sample_rate: int = 5,
        gamma: float = 1.0,
        faiss_backend: str = "auto",
        refinement: bool = True,
        refinement_dim: int = 30,
        refinement_n_neighbors: int = 30,
        refinement_n_epochs: Optional[int] = None,
        collect_loss: Union[bool, str] = False,
        keep_intermediate: bool = False,
        cleanup_after_fit: bool = True,
        _internal_allow_high_dim: bool = False,
        init_random_n_components: bool = False,
    ):
        # Main embedding hyperparameters
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.temperature = temperature
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.min_dist = min_dist
        self.spread = spread

        # Reproducibility and verbosity
        self.random_state = random_state
        self.deterministic = deterministic
        self.verbose = verbose

        # Hardware and batching options
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.torch_batch_size = torch_batch_size

        # Initialization and optimization options
        self.metric = metric
        self.init = init
        self.optimizer_backend = optimizer_backend
        self.negative_sample_rate = negative_sample_rate
        self.gamma = gamma
        self.faiss_backend = faiss_backend

        # Two-phase refinement options
        self.refinement = refinement
        self.refinement_dim = refinement_dim
        self.refinement_n_neighbors = refinement_n_neighbors
        self.refinement_n_epochs = refinement_n_epochs
        self.init_random_n_components = init_random_n_components

        # Diagnostics and memory options
        self.collect_loss = collect_loss
        self.keep_intermediate = keep_intermediate
        self.cleanup_after_fit = cleanup_after_fit
        self._internal_allow_high_dim = _internal_allow_high_dim

    def _default_epochs_for_n_samples(self, n_samples: int) -> int:
        """Return the same default epoch count used by optimization_simplicial_set."""
        return 500 if int(n_samples) <= 10000 else 200

    def _effective_first_epochs(self, n_samples: int) -> int:
        """Epoch count for phase 1."""
        if self.n_epochs is None:
            return self._default_epochs_for_n_samples(n_samples)
        return int(self.n_epochs)

    def _effective_second_epochs(self, first_epochs: int) -> int:
        """Automatic phase-2 epoch count: default is one quarter of phase 1."""
        if self.refinement_n_epochs is not None:
            return int(self.refinement_n_epochs)
        return max(1, int(round(float(first_epochs) / 4.0)))

    def _should_collect_loss(self, phase: str) -> bool:
        """Decide whether to collect optimization loss for a phase."""
        value = self.collect_loss
        if isinstance(value, bool):
            return bool(value)
        value = str(value).lower()
        if value in {"none", "false", "no", "0"}:
            return False
        if value in {"both", "true", "yes", "1", "all"}:
            return True
        if phase == "phase1" and value in {"phase1", "first", "30d"}:
            return True
        if phase == "phase2" and value in {"phase2", "second", "2d", "refinement"}:
            return True
        if phase == "single" and value in {"single", "main"}:
            return True
        return False

    def _copy_for_phase(
        self,
        *,
        n_components: int,
        n_neighbors: int,
        n_epochs: int,
        init: Union[str, np.ndarray],
        metric: Optional[str] = None,
        collect_loss: Union[bool, str],
    ) -> "CosMAP":
        """Create one internal CosMAP object for one refinement phase."""
        return CosMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            temperature=self.temperature,
            n_epochs=n_epochs,
            learning_rate=self.learning_rate,
            min_dist=self.min_dist,
            spread=self.spread,
            random_state=self.random_state,
            deterministic=self.deterministic,
            verbose=self.verbose,
            use_gpu=self.use_gpu,
            batch_size=self.batch_size,
            metric=self.metric if metric is None else metric,
            init=init,
            optimizer_backend=self.optimizer_backend,
            torch_batch_size=self.torch_batch_size,
            negative_sample_rate=self.negative_sample_rate,
            gamma=self.gamma,
            faiss_backend=self.faiss_backend,
            refinement=False,
            refinement_dim=self.refinement_dim,
            init_random_n_components=self.init_random_n_components,
            refinement_n_neighbors=self.refinement_n_neighbors,
            refinement_n_epochs=self.refinement_n_epochs,
            collect_loss=collect_loss,
            keep_intermediate=False,
            cleanup_after_fit=True,
            _internal_allow_high_dim=True,
        )

    def _fit_refinement_pipeline(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CosMAP":
        """Run the two-phase CosMAP refinement pipeline."""

        if int(self.refinement_dim) <= int(self.n_components):
            raise ValueError("refinement_dim must be larger than final n_components.")
        if int(self.refinement_n_neighbors) <= 0:
            raise ValueError("refinement_n_neighbors must be positive.")

        n_samples = int(X.shape[0])
        phase1_epochs = self._effective_first_epochs(n_samples)
        phase2_epochs = self._effective_second_epochs(phase1_epochs)

        self.phase1_n_epochs_ = phase1_epochs
        self.phase2_n_epochs_ = phase2_epochs

        if self.verbose:
            print(f"{ts()} Starting two-phase CosMAP refinement")
            print(
                f"{ts()} Phase 1: input {X.shape[1]}D -> {self.refinement_dim}D "
                f"| n_neighbors={self.n_neighbors} | n_epochs={phase1_epochs}"
            )

        phase1 = self._copy_for_phase(
            n_components=int(self.refinement_dim),
            n_neighbors=int(self.n_neighbors),
            n_epochs=phase1_epochs,
            init=self.init,
            collect_loss="single" if self._should_collect_loss("phase1") else False,
        )

        start_phase1 = time.time()
        X_embedded_high = phase1.fit_transform(X).astype(np.float32, copy=False)
        self.phase1_time_ = time.time() - start_phase1

        self._a_phase1_ = phase1._a
        self._b_phase1_ = phase1._b
        self.effective_optimizer_backend_phase1_ = phase1.effective_optimizer_backend_
        self.loss_history_phase1_ = getattr(phase1, "aux_data_", {}).get("loss_history", None)

        if self.verbose:
            print(f"{ts()} Phase 1 completed in {self.phase1_time_:.2f} seconds")
            print(
                f"{ts()} Phase 2: {self.refinement_dim}D -> {self.n_components}D "
                f"| n_neighbors={self.refinement_n_neighbors} | n_epochs={phase2_epochs}"
            )

        if self.init_random_n_components:
            rng, _ = make_random_state(self.random_state)
            random_dims = rng.choice(
                X_embedded_high.shape[1],
                size=int(self.n_components),
                replace=False,
            )
            if self.verbose:
                print(f"[CosMAP] Random init dimensions selected: {random_dims}")
            init_second = np.asarray(X_embedded_high[:, random_dims], dtype=np.float32, order="C")
        else:
            if self.verbose:
                print(f"[CosMAP] Using first {int(self.n_components)} dimensions for init.")
            init_second = np.asarray(X_embedded_high[:, : int(self.n_components)], dtype=np.float32, order="C")

        if self.keep_intermediate:
            self.phase1_model_ = phase1
            self.embedding_30d_ = X_embedded_high
            self.graph_phase1_ = phase1.graph_
        else:
            del phase1
            cleanup_torch_memory(verbose=self.verbose, label="after phase 1")
        
        phase2_metric = "euclidean" if str(self.metric).lower() == "precomputed" else self.metric

        phase2 = self._copy_for_phase(
            n_components=int(self.n_components),
            n_neighbors=int(self.refinement_n_neighbors),
            n_epochs=phase2_epochs,
            init=init_second,
            collect_loss="single" if self._should_collect_loss("phase2") else False,
            metric=phase2_metric,
        )

        start_phase2 = time.time()
        X_embedded_final = phase2.fit_transform(X_embedded_high).astype(np.float32, copy=False)
        self.phase2_time_ = time.time() - start_phase2

        self.embedding_ = X_embedded_final
        self.graph_ = phase2.graph_
        self._a = phase2._a
        self._b = phase2._b
        self.effective_optimizer_backend_phase2_ = phase2.effective_optimizer_backend_
        self.effective_optimizer_backend_ = phase2.effective_optimizer_backend_
        self.loss_history_phase2_ = getattr(phase2, "aux_data_", {}).get("loss_history", None)

        self.aux_data_ = {
            "phase1_time": self.phase1_time_,
            "phase2_time": self.phase2_time_,
            "phase1_n_epochs": phase1_epochs,
            "phase2_n_epochs": phase2_epochs,
            "phase1_backend": self.effective_optimizer_backend_phase1_,
            "phase2_backend": self.effective_optimizer_backend_phase2_,
            "keep_intermediate": bool(self.keep_intermediate),
        }

        if self.keep_intermediate:
            self.phase2_model_ = phase2
            self.graph_phase2_ = phase2.graph_
        else:
            del phase2

        if not self.keep_intermediate:
            del X_embedded_high
            del init_second

        if self.cleanup_after_fit:
            cleanup_torch_memory(verbose=self.verbose, label="after two-phase CosMAP")

        if self.verbose:
            print(f"{ts()} Phase 2 completed in {self.phase2_time_:.2f} seconds")
            total_time = self.phase2_time_ + self.phase1_time_
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            print(f"{ts()} All phase of  CosMAP completed  in {minutes:.2f} minutes and {seconds:.2f} seconds.")
            print(f"{ts()} Two-phase CosMAP completed")

        return self

    def _compute_similarity_matrix(self, X: np.ndarray):
        """Compute CosMAP's sparse temperature-scaled similarity graph."""

        return compute_similarity_graph(
                    X,
                    metric=self.metric,
                    n_neighbors=int(self.n_neighbors),
                    temperature=float(self.temperature),
                    use_gpu=bool(self.use_gpu),
                    batch_size=int(self.batch_size),
                    faiss_backend=str(self.faiss_backend),
                    verbose=bool(self.verbose),
                )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CosMAP":
        """Fit CosMAP to X."""
        X = check_array(X, accept_sparse=False, dtype=np.float32)

        if bool(self.refinement):
            return self._fit_refinement_pipeline(X, y)

        if int(self.n_components) > 3 and not bool(getattr(self, "_internal_allow_high_dim", False)):
            raise ValueError(
                "When refinement=False, n_components must be 2 or 3 at most. "
                "Use refinement=True with refinement_dim=30 for the two-phase pipeline."
            )

        if self.verbose:
            print(f"{ts()} Computing CosMAP similarity graph")
        self.graph_ = self._compute_similarity_matrix(X)

        if self.verbose:
            print(f"{ts()} Fitting low-dimensional kernel parameters")
        a, b = find_ab_(self.spread, self.min_dist)
        self._a = a
        self._b = b

        rng, seed = make_random_state(self.random_state)
        parallel = self.random_state is None


 

        effective_optimizer_backend = str(self.optimizer_backend).lower()

        # Torch optimizer can run on CUDA NVIDIA or MPS.
        # on MacBook, CUDA is unavailable but MPS may be available .
        torch_gpu_available = (
            bool(self.use_gpu)
            and (
                torch.cuda.is_available()
                or (
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_built()
                    and torch.backends.mps.is_available()
                )
            )
        )

        if effective_optimizer_backend in {"torch_manual", "torch_autograd"} and not torch_gpu_available:
            if self.verbose:
                print(
                    f"{ts()} No CUDA or MPS device is available; switching layout optimizer "
                    f"from '{effective_optimizer_backend}' to 'cpu numba manual layout optimization'."
                )
            effective_optimizer_backend = "cpu_numba_manual" #or "umap" if you want to use the original UMAP optimizer as fallback, but it is not optimized for high-dimensional data for CosMAP and may be very slow. .


        # Do NOT fall back to CPU/UMAP when GPU is available.  The selected
        # torch optimizer uses get_torch_device(use_gpu), so CUDA is used when
        # available.  If no GPU exists,  it explicitly choose optimizer_backend='umap' 
        
        device = get_torch_device(use_gpu=self.use_gpu)
        if self.verbose:
            print(f"{ts()} Selected torch device: {device}")

        self.effective_optimizer_backend_ = effective_optimizer_backend

        if self.verbose:
            print(f"{ts()} Optimizing embedding with backend='{effective_optimizer_backend}'")

        self.embedding_, self.aux_data_ = optimization_simplicial_set(
            data=X,
            graph=self.graph_,
            n_components=self.n_components,
            initial_alpha=self.learning_rate,
            a=a,
            b=b,
            gamma=self.gamma,
            negative_sample_rate=self.negative_sample_rate,
            n_epochs=self.n_epochs,
            init=self.init,
            random_state=rng,
            seed=seed,
            deterministic=bool(self.deterministic),
            metric=self.metric,
            metric_kwds={},
            densmap=False, #We do not implementated the Dense CosMAP
            densmap_kwds={},
            output_dens=False,
            output_metric="euclidean", # Only euclidean output metric for now
            output_metric_kwds={},
            euclidean_output=True,
            parallel=parallel,
            verbose=self.verbose,
            tqdm_kwds=None,
            optimizer_backend=effective_optimizer_backend,
            use_gpu=self.use_gpu,
            torch_batch_size=self.torch_batch_size,
            collect_loss=self._should_collect_loss("single"),
        )

        if self.cleanup_after_fit:
            cleanup_torch_memory(verbose=self.verbose, label="after single-phase CosMAP")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data. Not implemented yet."""
        check_is_fitted(self, ["embedding_", "graph_"])
        raise NotImplementedError(
            "Transform for new data points is not yet implemented. "
            "Use fit_transform for the training data."
        )

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return the embedding."""
        return self.fit(X, y).embedding_
