
"""
Author: Fenosoa Randrianjatovo 
CosMAP benchmark pipeline.

This file is designed to be used directly in a notebook or as a Python module.


------------------------
The CNE examples from the official contrastive-ne repository use:

    InfoNC-t-SNE:
        model = cne.CNE()
        emb = model.fit_transform(X.astype(float))

    Neg-t-SNE:
        model = cne.CNE(loss_mode="neg")
        emb = model.fit_transform(X.astype(float))

Therefore, this script does NOT pass a custom PCA initialization to CNE-based
InfoNC-t-SNE or Neg-t-SNE. The benchmark keeps internal method keys such as
"infonce_2d" and "negtsne_2d", but displays them as "InfoNC-$t$-SNE" and
"Neg-$t$-SNE" in plots.

"""

from __future__ import annotations

import importlib
import os
import time
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Small utility functions
# ---------------------------------------------------------------------


def _import_class(module_name: str, class_name: str):
    """
    Dynamically import a class from a module.

    Example
    -------
    _import_class("umap", "UMAP") returns umap.UMAP
    _import_class("sklearn.decomposition", "PCA") returns sklearn.decomposition.PCA
    """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)



def _ensure_2d_numpy(X: Any, name: str = "X") -> np.ndarray:
    """Convert input to NumPy and ensure it is 2-dimensional."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {X.shape}")
    return X



def _ensure_1d_numpy(y: Any, name: str = "y") -> np.ndarray:
    """Convert input labels to NumPy and ensure they are 1-dimensional."""
    y = np.asarray(y)
    if y.ndim != 1:
        y = np.ravel(y)
    return y



def _safe_np_save(path: str, obj: Any) -> None:
    """
    Save a Python object or array with NumPy.

    Dictionaries are saved as object arrays and should be loaded with:
        np.load(path, allow_pickle=True).item()
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, obj, allow_pickle=True)



def _load_dataset_from_npy(data_name: str, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load X and y from .npy files.

    Expected file names, in this order:
        1. {data_name}.npy and {data_name}_labels.npy
        2. X.npy and y.npy
        3. X_{data_name}.npy and labels_{data_name}.npy

    Parameters
    ----------
    data_name:
        Dataset name, for example "paul15".

    data_path:
        Folder containing the .npy files.
    """
    data_path = os.path.abspath(data_path)

    candidates = [
        (
            os.path.join(data_path, f"{data_name}.npy"),
            os.path.join(data_path, f"{data_name}_labels.npy"),
        ),
        (
            os.path.join(data_path, "X.npy"),
            os.path.join(data_path, "y.npy"),
        ),
        (
            os.path.join(data_path, f"X_{data_name}.npy"),
            os.path.join(data_path, f"labels_{data_name}.npy"),
        ),
    ]

    for x_path, y_path in candidates:
        if os.path.exists(x_path) and os.path.exists(y_path):
            X = np.load(x_path, allow_pickle=True)
            X=X.reshape(X.shape[0],-1)
            y = np.load(y_path, allow_pickle=True)
            return _ensure_2d_numpy(X, "X"), _ensure_1d_numpy(y, "y")

    searched = "\n".join([f"  - {x}\n    {y}" for x, y in candidates])
    raise FileNotFoundError(
        f"Could not find dataset files for data_name={data_name!r} in {data_path}.\n"
        f"Searched:\n{searched}"
    )



def _fit_transform_model(
    model: Any,
    X: np.ndarray,
    fit_transform_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Generic fit_transform wrapper for most DR methods.

    This is used for methods that do not require special arguments, while
    still allowing a method-specific fit_transform argument when needed.
    Example: NCVis uses fit_transform(X, init="pca").
    """
    if fit_transform_kwargs is None:
        fit_transform_kwargs = {}

    if hasattr(model, "fit_transform"):
        emb = model.fit_transform(X, **fit_transform_kwargs)
        # emb = model.fit_transform(X)


        
    else:
        if fit_transform_kwargs:
            raise ValueError(
                f"Model {model.__class__.__name__} does not have fit_transform, "
                f"so fit_transform_kwargs={fit_transform_kwargs} cannot be used."
            )

        model.fit(X)
        if hasattr(model, "transform"):
            emb = model.transform(X)
        elif hasattr(model, "embedding_"):
            emb = model.embedding_
        else:
            raise AttributeError(
                f"Model {model.__class__.__name__} has no fit_transform, transform, or embedding_."
            )

    return np.asarray(emb)



def _sanitize_cosmap_params(cosmap_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Guard against the common typo:

        COSMAP_PARAMS = {...},

    The trailing comma makes it a tuple containing one dictionary.
    This helper converts ({...},) back to {...} and raises a clear error
    for unsupported types.
    """
    if cosmap_params is None:
        return None

    if isinstance(cosmap_params, tuple):
        if len(cosmap_params) == 1 and isinstance(cosmap_params[0], dict):
            return cosmap_params[0]
        raise TypeError(
            "cosmap_params is a tuple. Did you add a trailing comma after the dict? "
            "Expected a dict like COSMAP_PARAMS = {...}, without the final comma."
        )

    if not isinstance(cosmap_params, dict):
        raise TypeError(f"cosmap_params must be a dict or None, got {type(cosmap_params)}")

    return cosmap_params




# ---------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------


def build_method_registry(
    random_state: int = 42,
    n_components: int = 2,
    cosmap_params: Optional[Dict[str, Any]] = None,
    umap_params: Optional[Dict[str, Any]] = None,
    umap_metric: str = "euclidean",
    pacmap_metric: str = "euclidean",
) -> Dict[str, Dict[str, Any]]:
    """
    Build a method dictionary similar to your previous bench_mark.py.

    Keys are intentionally named like cosmap_2d, umap_2d, pacmap_2d, etc.

    Important CNE implementation
    ----------------------------
    InfoNC-t-SNE is implemented as cne.CNE().
    Neg-t-SNE is implemented as cne.CNE(loss_mode="neg").

    Both are called as:
        model.fit_transform(X.astype(float))

    following the official contrastive-ne examples.
    """

    cosmap_params = _sanitize_cosmap_params(cosmap_params)

    default_cosmap_params: Dict[str, Any] = {
                "n_components": n_components,
                "n_neighbors": 15,
                "temperature": 0.5,
                "n_epochs": None,
                "random_state": random_state,
                "deterministic": False,
                "verbose": True,
                "use_gpu": 0,
                "metric": "cosine",
    }
    if cosmap_params:
        default_cosmap_params.update(cosmap_params)

        
    default_umap_params: Dict[str, Any] = {
        "n_components": n_components,
        "n_neighbors": 15,
        "random_state": random_state,
        "verbose": True,
        "metric": "euclidean",
        "min_dist": 0.1,
        "spread": 1.0,
        "n_epochs": None,
        
    }
    
    if umap_params:
        default_umap_params.update(umap_params)

    return {
        "cosmap_2d": {
            "module": "cosmapdr",
            "class": "CosMAP",
            "params": default_cosmap_params,
            "astype_float": False,
        },
        "pacmap_2d": {
            "module": "pacmap",
            "class": "PaCMAP",
            "params": {
                "n_components": n_components,
                "random_state": random_state,
            },
            "astype_float": False,
        },
        "localmap_2d": {
            "module": "pacmap",
            "class": "LocalMAP",
            "params": {
                "n_components": n_components,
                "random_state": random_state,
            },
            "astype_float": False,
        },
        "tsne_2d": {
            "module": "sklearn.manifold",
            "class": "TSNE",
            "params": {
                "n_components": n_components,
                "random_state": random_state,
                "init": "pca",
                "learning_rate": "auto",
            },
            "astype_float": False,
        },
        "umap_2d": {
            "module": "umap",
            "class": "UMAP",
            "params": default_umap_params,
            "astype_float": False,
        },
        "hnne_2d": {
            "module": "hnne",
            "class": "HNNE",
            "params": {},
            "astype_float": False,
        },

        # CNE family ---------------------------------------------------
        # Official contrastive-ne examples:
        #   InfoNC-t-SNE: cne.CNE().fit_transform(X.astype(float))
        #   Neg-t-SNE:   cne.CNE(loss_mode="neg").fit_transform(X.astype(float))
        "infonce_2d": {
            "module": "cne",
            "class": "CNE",
            "params": {},
            "astype_float": True,
        },
        "negtsne_2d": {
            "module": "cne",
            "class": "CNE",
            "params": {"loss_mode": "neg"},
            "astype_float": True,
        },
        "ncvi_2d": {
            "module": "cne",
            "class": "CNE",
            "params": {
                "loss_mode": "nce",
                "optimizer": "adam",
                "parametric": True,
            },
            "astype_float": True,
            "fit_transform_kwargs": {
            "init": "pca"
    },
        },

        "trimap_2d": {
            "module": "trimap",
            "class": "TRIMAP",
            "params": {},
            "astype_float": False,
        },
        "phate_2d": {
            "module": "phate",
            "class": "PHATE",
            "params": {"n_jobs": -1},
            "astype_float": False,
        },
        "pca_2d": {
            "module": "sklearn.decomposition",
            "class": "PCA",
            "params": {
                "n_components": n_components,
                "random_state": random_state,
            },
            "astype_float": False,
        },
    }


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------


METHOD_DISPLAY_NAMES: Dict[str, str] = {
    "cosmap_2d": "CosMAP",
    "umap_2d": "UMAP",
    "pacmap_2d": "PaCMAP",
    "localmap_2d": "LocalMAP",
    "tsne_2d": "t-SNE",
    "trimap_2d": "TriMap",
    "phate_2d": "PHATE",
    "pca_2d": "PCA",
    "hnne_2d": "h-NNE",
    "infonce_2d": r"InfoNC-$t$-SNE",
    "negtsne_2d": r"Neg-$t$-SNE",
    "ncvi_2d": "NCVis",
}


def get_method_display_name(method_name: str) -> str:
    """Return the clean display name used in plots."""
    return METHOD_DISPLAY_NAMES.get(method_name, method_name.replace("_2d", ""))



def plot_single_embedding(
    embedding: np.ndarray,
    y: np.ndarray,
    method_name: str,
    dataset_name: str = "",
    s: float = 1.0,
    alpha: float = 0.7,
    cmap: str = "tab10",
    show_legend: bool = False,
    figsize: Tuple[float, float] = (8, 6),
):
    """
    Plot one 2D embedding.

    This function uses the corrected display names:
        infonce_2d -> InfoNC-$t$-SNE
        negtsne_2d -> Neg-$t$-SNE
    """
    import matplotlib.pyplot as plt

    embedding = np.asarray(embedding)
    y = _ensure_1d_numpy(y, "y")
    display_name = get_method_display_name(method_name)

    plt.figure(figsize=figsize)

    if show_legend:
        unique_labels = np.unique(y)
        for label in unique_labels:
            idx = np.where(y == label)[0]
            plt.scatter(
                embedding[idx, 0],
                embedding[idx, 1],
                label=str(label),
                s=s,
                alpha=alpha,
                edgecolor="none",
            )
        plt.legend(markerscale=4, fontsize=8)
    else:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=y,
            s=s,
            alpha=alpha,
            cmap=cmap,
            edgecolor="none",
        )

    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.axis("off")

    if dataset_name:
        plt.title(f"{display_name} of {dataset_name}")
    else:
        plt.title(display_name)

    plt.show()



def plot_embeddings_grid(
    embeddings_dict: Dict[str, Any],
    y: Optional[np.ndarray] = None,
    data_name: str = "",
    methods: Optional[Iterable[str]] = None,
    s: float = 1.0,
    alpha: float = 0.7,
    cmap: str = "tab10",
    ncols: int = 4,
    figsize_per_panel: Tuple[float, float] = (4, 4),
):
    """
    Plot several embeddings from embeddings_dict in a grid.

    Parameters
    ----------
    embeddings_dict:
        Dictionary returned by run_dimensionality_reduction_benchmark.

    y:
        Labels. If None, this function tries embeddings_dict["labels"].

    data_name:
        Dataset name used in the figure title.

    methods:
        Method keys to plot. If None, all available 2D method keys are plotted.
    """
    import math
    import matplotlib.pyplot as plt

    if y is None:
        if "labels" not in embeddings_dict:
            raise ValueError("Please provide y or include embeddings_dict['labels'].")
        y = embeddings_dict["labels"]
    y = _ensure_1d_numpy(y, "y")

    if methods is None:
        methods = [k for k, v in embeddings_dict.items()
                   if k.endswith("_2d") and np.asarray(v).ndim == 2]
    methods = list(methods)

    n = len(methods)
    if n == 0:
        raise ValueError("No embedding methods found to plot.")

    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    for ax, method_name in zip(axes.ravel(), methods):
        emb = np.asarray(embeddings_dict[method_name])
        ax.scatter(
            emb[:, 0],
            emb[:, 1],
            c=y,
            s=s,
            alpha=alpha,
            cmap=cmap,
            edgecolor="none",
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
        ax.set_title(get_method_display_name(method_name))

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if data_name:
        fig.suptitle(f"Dimensionality reduction embeddings on {data_name}", fontsize=14)

    plt.show()


# ---------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------


def run_dimensionality_reduction_benchmark(
    data_name: str,
    random_state: int = 42,
    data_path: Optional[str] = None,
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
    methods_to_run: Optional[Iterable[str]] = None,
    cosmap_params: Optional[Dict[str, Any]] = None,
    umap_params: Optional[Dict[str, Any]] = None,
    method_params_override: Optional[Dict[str, Dict[str, Any]]] = None,
    n_components: int = 2,
    umap_metric: str = "euclidean", #default
    pacmap_metric: str = "euclidean",
    save_individual_files: bool = False,
    save_failed_errors: bool = True,
    overwrite: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Run DR methods and save results 

    Output dictionary format
    ------------------------
    embeddings_dict[f"X_{data_name}"] = X
    embeddings_dict[f"labels_{data_name}"] = y
    embeddings_dict["cosmap_2d"] = embedding array, shape (n_samples, 2)
    embeddings_dict["umap_2d"] = embedding array, shape (n_samples, 2)

    Timing dictionary format
    ------------------------
    timing_dict["cosmap_2d"] = seconds
    timing_dict["umap_2d"] = seconds

    Returns
    -------
    embeddings_dict, timing_dict, output_folder
    """

    # -----------------------------
    # Data loading
    # -----------------------------
    if X is None or y is None:
        if data_path is None:
            data_path = "."
        if verbose:
            print(f"Loading {data_name} from {data_path}")
        X, y = _load_dataset_from_npy(data_name, data_path)
        X=X.reshape(X.shape[0], -1)  # Flatten if needed, but keep 2D shape
    else:
        X = _ensure_2d_numpy(X, "X")
        y = _ensure_1d_numpy(y, "y")

    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have different lengths: X={X.shape}, y={y.shape}")

    if verbose:
        print(f" Dataset: {data_name}")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        print(f"   unique labels: {len(np.unique(y))}")

    # -----------------------------
    # Output folder
    # -----------------------------
    if output_dir is None:
        # stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_folder = f"Cosmap_comparison_{data_name}_seed_{random_state}_{stamp}"
        output_folder = f"Output_comparison"
    else:
        output_folder = output_dir

    os.makedirs(output_folder, exist_ok=True)
    individual_dir = os.path.join(output_folder, "individual_embeddings")
    if save_individual_files:
        os.makedirs(individual_dir, exist_ok=True)
        
    # stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    embeddings_path = os.path.join(output_folder, f"all_embeddings_dict_{random_state}_{data_name}.npy")
    timing_path = os.path.join(output_folder, f"timing_dict_{random_state}_{data_name}.npy")
    errors_path = os.path.join(output_folder, f"errors_dict_{random_state}_{data_name}.npy")

    if (not overwrite) and os.path.exists(embeddings_path):
        if verbose:
            print(f"Existing result found and overwrite=False: {embeddings_path}")
            print("   Loading existing dictionaries instead of recomputing.")
        embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
        timing_dict = np.load(timing_path, allow_pickle=True).item() if os.path.exists(timing_path) else {}
        return embeddings_dict, timing_dict, output_folder

    # -----------------------------
    #  Methods
    # -----------------------------
    all_methods = build_method_registry(
        random_state=random_state,
        n_components=n_components,
        cosmap_params=cosmap_params,
        umap_params=umap_params,
        umap_metric=umap_metric,
        pacmap_metric=pacmap_metric,
    )

    if method_params_override:
        for method_name, extra_params in method_params_override.items():
            if method_name not in all_methods:
                raise KeyError(f"Unknown method in method_params_override: {method_name}")
            all_methods[method_name]["params"].update(extra_params)

    if methods_to_run is None:
        methods = all_methods
    else:
        methods_to_run = list(methods_to_run)
        missing = [m for m in methods_to_run if m not in all_methods]
        if missing:
            raise KeyError(f"Unknown methods: {missing}. Available: {list(all_methods.keys())}")
        methods = {m: all_methods[m] for m in methods_to_run}

    if verbose:
        print(f" Methods to run: {list(methods.keys())}")
        print(f"Output folder: {output_folder}")

    # -----------------------------
    # Storage compatible
    # -----------------------------
    embeddings_dict: Dict[str, Any] = {
        f"X_{data_name}": X,
        f"labels_{data_name}": y,
        "X": X,
        "labels": y,
    }
    timing_dict: Dict[str, float] = {}
    errors_dict: Dict[str, str] = {}

    # -----------------------------
    # Run methods
    # -----------------------------
    iterator = tqdm(methods.items(), total=len(methods), disable=not verbose, desc="Running DR")

    for method_name, cfg in iterator:
        iterator.set_description(f"Running {method_name}")
        start_time = time.time()

        try:
            ModelClass = _import_class(cfg["module"], cfg["class"])
            model = ModelClass(**cfg["params"])

            # CNE official examples use X.astype(float).
            # For non-CNE methods we keep the original input unchanged.
            X_input = X.astype(float) if cfg.get("astype_float", False) else X

            fit_transform_kwargs = cfg.get("fit_transform_kwargs", {})
            emb = _fit_transform_model(model, X_input, fit_transform_kwargs=fit_transform_kwargs)
            # emb = _fit_transform_model(model, X_input)

            emb = np.asarray(emb)
            total_time = time.time() - start_time

            embeddings_dict[method_name] = emb
            timing_dict[method_name] = total_time

            if save_individual_files:
                ind_path = os.path.join(
                    individual_dir,
                    f"{data_name}_{method_name}_seed_{random_state}.npy",
                )
                _safe_np_save(ind_path, emb)

            if verbose:
                tqdm.write(f"✅ {method_name:15} | shape={emb.shape} | time={total_time:.2f}s")

        except Exception as e:
            total_time = time.time() - start_time
            timing_dict[method_name] = np.nan
            errors_dict[method_name] = repr(e)
            if verbose:
                tqdm.write(f"❌ {method_name:15} | failed after {total_time:.2f}s | {repr(e)}")
            continue

    # -----------------------------
    # Save results
    # -----------------------------
    _safe_np_save(embeddings_path, embeddings_dict)
    _safe_np_save(timing_path, timing_dict)
    if save_failed_errors:
        _safe_np_save(errors_path, errors_dict)

    # -----------------------------
    # Summary
    # -----------------------------
    successful = [k for k in methods if k in embeddings_dict]
    failed = [k for k in methods if k not in embeddings_dict]

    if verbose:
        print("\n" + "=" * 72)
        print("BENCHMARK SUMMARY")
        print("=" * 72)
        print(f"Dataset: {data_name}")
        print(f"Seed: {random_state}")
        print(f"Successful methods: {len(successful)}/{len(methods)}")
        print(f"Failed methods: {len(failed)}/{len(methods)}")

        print("\nAvailable embeddings:")
        for method_name in successful:
            emb = embeddings_dict[method_name]
            sec = timing_dict.get(method_name, np.nan)
            print(f"  • {method_name:15} | shape={str(emb.shape):14} | time={sec:8.2f}s")

        if failed:
            print("\nFailed methods:")
            for method_name in failed:
                print(f"  • {method_name}: {errors_dict.get(method_name, 'unknown error')}")

        print("\nSaved files:")
        print(f"  • {embeddings_path}")
        print(f"  • {timing_path}")
        if save_failed_errors:
            print(f"  • {errors_path}")

        print("\nNotebook loading example:")
        print(f"  embeddings_dict = np.load('{embeddings_path}', allow_pickle=True).item()")
        print(f"  timing_dict = np.load('{timing_path}', allow_pickle=True).item()")
        if "cosmap_2d" in successful:
            print("  cosmap_embedding = embeddings_dict['cosmap_2d']")

    return embeddings_dict, timing_dict, output_folder

