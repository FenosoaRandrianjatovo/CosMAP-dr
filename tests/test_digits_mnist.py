import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits

from cosmapdr import CosMAP, diagnose_cosmap_environment


def test_cosmap_digits(tmp_path):
    """Verify that CosMAP produces a valid embedding without network access."""

    digits = load_digits()
    X = digits.data
    y = digits.target.astype(int)

    n_subset = min(100, X.shape[0])

    rng = np.random.default_rng(seed=42)
    idx = rng.choice(X.shape[0], size=n_subset, replace=False)

    X = X[idx]
    y = y[idx]

    diagnose_cosmap_environment()

    model = CosMAP(
        n_components=2,
        n_neighbors=15,
        temperature=0.5,
        n_epochs=None,
        random_state=42,
        deterministic=False,
        verbose=True,
        use_gpu=0,
        metric="cosine",
    )

    embedding = model.fit_transform(X)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (n_subset, 2)
    assert np.isfinite(embedding).all()
    assert embedding.std() > 0

    figure_path = tmp_path / "cosmap_digits.png"

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
    ax.set_title("CosMAP embedding of sklearn digits")
    fig.savefig(figure_path, dpi=100)
    plt.close(fig)

    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
