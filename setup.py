from setuptools import setup, find_packages

setup(
    name="cosmap-dr",
    version="0.1.0",
    description="CosMAP: Contrastive Manifold Approximation and Projection",
    author="Fenosoa Randrianjatovo",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "tqdm",
        "umap-learn",
    ],
    extras_require={
        "gpu": ["faiss-gpu"],
        "dev": ["matplotlib", "pytest"],
    },
)
