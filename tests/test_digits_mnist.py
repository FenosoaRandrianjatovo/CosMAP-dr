import matplotlib

# Use a non-interactive backend for CI environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml

from cosmapdr import CosMAP, diagnose_cosmap_environment


def test_cosmap_mnist(tmp_path):
    """Verify that CosMAP can embed a small subset of MNIST dataset."""

    # Use a small subset to keep the automated test fast.
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
  
    n_subset = 1000 # For testing
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(X.shape[0], size=n_subset, replace=False)
    X = X[idx]
    y = y[idx]


    diagnose_cosmap_environment()

    cosmap_ = CosMAP(
                      n_components=2,
                      n_neighbors=15,
                      temperature=0.5,
                      n_epochs=None,
                      random_state=42,          # no fixed seed, faster stochastic path
                      deterministic=False,        # do not force slow deterministic CUDA kernels
                      verbose=True,
                      use_gpu=0,
                      metric="cosine",
                  )

    embedding = cosmap_.fit_transform(X)

    # Validate the returned embedding.
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (n_subset, 2)
    assert np.isfinite(embedding).all()
    assert embedding.std() > 0

    
    figure_path = tmp_path / "cosmap_MNIST.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=y,
        cmap="tab10",
        s=5,
        alpha=0.7,
    )
    fig.colorbar(scatter, ax=ax)
    ax.set_title("CosMAP embedding of MNIST")
    fig.savefig(figure_path, dpi=100)
    plt.close(fig)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
