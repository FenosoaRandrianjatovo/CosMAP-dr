import matplotlib

matplotlib.use("Agg")

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler

from cosmapdr import CosMAP, diagnose_cosmap_environment


def _run_cosmap_test(
    X: np.ndarray,
    y: np.ndarray,
    figure_path: Path,
    title: str,
    *,
    n_subset: Optional[int] = None,
    metric: str = "cosine",
    random_state: int = 422,
) -> np.ndarray:
    """Run CosMAP and validate the resulting embedding."""

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert np.isfinite(X).all()

    if n_subset is not None and n_subset < X.shape[0]:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(
            X.shape[0],
            size=n_subset,
            replace=False,
        )

        X = X[indices]
        y = y[indices]

    n_samples = X.shape[0]
    n_neighbors = min(15, n_samples - 1)

    model = CosMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        temperature=0.5,
        n_epochs=None,
        random_state=random_state,
        deterministic=False,
        verbose=False,
        use_gpu=0,
        metric=metric,
    )

    embedding = model.fit_transform(X)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (n_samples, 2)
    assert np.issubdtype(embedding.dtype, np.number)
    assert np.isfinite(embedding).all()
    assert embedding.std() > 0

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=10,
        alpha=0.7,
    )

    fig.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("CosMAP 1")
    ax.set_ylabel("CosMAP 2")
    fig.tight_layout()

    fig.savefig(figure_path, dpi=100)
    plt.close(fig)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0

    return embedding


def test_cosmap_digits(tmp_path):
    """Verify CosMAP on a higher-dimensional dataset."""

    digits = load_digits()

    X = digits.data
    y = digits.target

    assert X.shape[1] == 64

    diagnose_cosmap_environment()

    _run_cosmap_test(
        X=X,
        y=y,
        n_subset=100,
        metric="cosine",
        figure_path=tmp_path / "cosmap_digits.png",
        title="CosMAP embedding of sklearn digits",
    )


def test_cosmap_iris_low_dimensional_input(tmp_path):
    """Verify CosMAP when the input dimension is below five."""

    iris = load_iris()

    X = iris.data
    y = iris.target

    # Iris contains 150 observations and only four features.
    assert X.shape == (150, 4)
    assert X.shape[1] < 5

    # Standardization is useful because Iris features use different scales.
    X = StandardScaler().fit_transform(X)

    embedding = _run_cosmap_test(
        X=X,
        y=y,
        metric="euclidean",
        figure_path=tmp_path / "cosmap_iris.png",
        title="CosMAP embedding of sklearn Iris",
    )

    assert embedding.shape == (150, 2)
