from cosmap import CosMAP, diagnose_cosmap_environment
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

X, y = load_digits(return_X_y=True)

diagnose_cosmap_environment()

model = CosMAP(
    n_components=2,
    n_neighbors=15,
    temperature=0.5,
    n_epochs=50,
    random_state=None,
    deterministic=False,
    verbose=True,
    use_gpu=True,
    optimizer_backend="torch_manual",
    faiss_backend="auto",
)

emb = model.fit_transform(X)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
plt.colorbar(scatter)
plt.title("CosMAP embedding of sklearn digits")
plt.show()
