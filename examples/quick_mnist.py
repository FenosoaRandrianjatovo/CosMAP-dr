import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from cosmap import CosMAP, diagnose_cosmap_environment

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

diagnose_cosmap_environment()

cosmap = CosMAP(
    n_components=2,
    n_neighbors=15,
    temperature=0.5,
    n_epochs=200,
    random_state=None,
    deterministic=False,
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
