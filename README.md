# CosMAP

CosMAP (**Contrastive Manifold Approximation and Projection**) is a dimensionality-reduction estimator with a scikit-learn-like API. It builds a high-dimensional graph from cosine similarity and a temperature-scaled local softmax, then optimizes a low-dimensional embedding with a binary cross-entropy objective.


## Installation

### Prerequisites

- Python >= 3.7
- pip (Python package installer)

### From source (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/FenosoaRandrianjatovo/CosMAP-dr.git
cd CosMAP-dr
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

3. Install the package (From the project root):


```bash
pip install -e .
```
4. For Google Colab, use a normal installation instead:
   
```bash
import sys
!{sys.executable} -m pip uninstall -y cosmap-dr
!{sys.executable} -m pip install .
```

Activate your virtual environment, then run the same command.

## Check GPU / FAISS environment

```python
from cosmap import diagnose_cosmap_environment

diagnose_cosmap_environment()
```

CosMAP's policy is:

1. Use FAISS-GPU if CUDA and FAISS-GPU are available.
2. If CUDA is available but FAISS is CPU-only, skip FAISS-CPU and use batched torch GPU kNN.
3. Use sklearn CPU kNN only when no CUDA/MPS GPU device is selected or available.

## Quick run

```python
import numpy as np
from cosmap import CosMAP
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

cosmap = CosMAP(
    n_components=2,
    n_neighbors=15,
    temperature=0.5,
    n_epochs=200,
    random_state=None,          # no fixed seed, faster stochastic path
    deterministic=False,        # do not force slow deterministic CUDA kernels
    verbose=True,
    use_gpu=True,
    optimizer_backend="torch_manual",
    faiss_backend="auto",
)

X_embedded = cosmap.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="tab10", s=1, alpha=0.7)
plt.colorbar(scatter)
plt.title("CosMAP embedding of MNIST dataset")
plt.show()
```

## Benchmarks

To evaluate the visual quality of CosMAP embeddings, we applied the method to two standard handwritten digit datasets: **MNIST** and **USPS**. These datasets are widely used benchmarks in dimensionality reduction and manifold learning because they contain multiple visually similar classes, making cluster separation a non-trivial task.

The results below show that CosMAP produces well-structured two-dimensional embeddings with clearly separated digit clusters. In both datasets, samples belonging to the same digit tend to form compact neighborhoods, while different digit classes are projected into distinct regions of the embedding space. This suggests that CosMAP is able to preserve meaningful local relationships while maintaining a globally interpretable organization of the data.

Compared with other dimensionality reduction methods, CosMAP provides visually sharper cluster boundaries and improved separation of true classes, particularly in regions where digit shapes are naturally ambiguous.


<!-- <h3 align="center"></h3> -->

### MNIST

<p align="center">
  <img src="https://github.com/FenosoaRandrianjatovo/CosMAP-dr/blob/main/images/mnist.png" alt="CosMAP visualization of MNIST" >
</p>

<p align="center">
  <strong>Figure 1. CosMAP embedding of the MNIST dataset.</strong><br>
    
  Two-dimensional visualization of the MNIST handwritten digit dataset containing <strong>70,000 samples</strong> and <strong>784 features</strong>. Each point represents one image, and colors correspond to the true digit labels.
</p>




<!-- <h3 align="center">MNIST</h3> -->

### USPS

<p align="center">
  <img src="https://github.com/FenosoaRandrianjatovo/CosMAP-dr/blob/main/images/usps.png" alt="CosMAP visualization of MNIST">
</p>

<p align="center">
  <strong>Figure 2. CosMAP embedding of the USPS dataset.</strong><br>



Two-dimensional visualization of the USPS handwritten digit dataset containing **9,298 samples** and **256 features**. Despite the smaller image resolution and the higher visual similarity between some digit classes, CosMAP still produces a well-organized embedding with clearly distinguishable clusters. This demonstrates the robustness of the method across different handwritten digit datasets.

## Reproducibility modes

Fast stochastic mode:

```python
CosMAP(random_state=None, deterministic=False)
```

Seeded but not strict deterministic GPU mode:

```python
CosMAP(random_state=42, deterministic=False)
```

Strict deterministic mode for experiments where exact repeatability is more important than speed:

```python
CosMAP(random_state=42, deterministic=True)
```


## Parameters

- `n_components`: Number of dimensions in the embedded space (default: 2)
- `n_neighbors`: Number of nearest neighbors to consider (default: 15)
- `temperature`: Temperature parameter for cosine similarity (default: 0.5)
- `n_epochs`: Number of optimization epochs (default: 500)
- `learning_rate`: Initial learning rate (default: 1.0)
- `min_dist`: Minimum distance between points in embedding (default: 0.1)
- `spread`: Scale of embedded points (default: 1.0)
- `random_state`: Random seed for reproducibility (default: None)
- `verbose`: Whether to print progress information (default: False)
- `use_gpu`: Whether to use GPU acceleration if available (default: True)

## API Compatibility

CosMAP follows the scikit-learn API conventions:

```python
# Similar to UMAP, PaCMAP, t-SNE
from cosmap import CosMAP
from umap import UMAP
from sklearn.manifold import TSNE

# All have the same interface
cosmap = CosMAP(n_components=2)
umap = UMAP(n_components=2)
tsne = TSNE(n_components=2)

# Fit and transform
X_cosmap = cosmap.fit_transform(X)
X_umap = umap.fit_transform(X)
X_tsne = tsne.fit_transform(X)
```

## Requirements

- Python >= 3.7
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- scikit-learn >= 0.24.0
- PyTorch >= 1.7.0
- umap-learn >= 0.5.0
- matplotlib >= 3.3.0
- tqdm >= 4.50.0

Optional:

- FAISS (for faster GPU-accelerated nearest neighbor search)

## License

BSD 2-Clause License


