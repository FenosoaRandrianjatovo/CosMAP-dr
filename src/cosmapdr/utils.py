# Author: Fenosoa  Randrianjatovo <RandrianjatovoFenosoa@gmail.com>
"""General utilities for CosMAP."""
from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def ts() -> str:
    """Small timestamp helper for verbose messages."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def get_torch_device(use_gpu: bool = True) -> torch.device:
    """Return CUDA, then MPS, then CPU depending on availability."""
    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if (
        use_gpu
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def cleanup_torch_memory(verbose: bool = False, label: str = "") -> None:
    """Clean Python, CUDA, and MPS memory after a heavy CosMAP step."""
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            prefix = f"{ts()} [{label}] " if label else f"{ts()} "
            print(
                f"{prefix}CUDA memory after cleanup: "
                f"allocated={allocated:.3f} GB, reserved={reserved:.3f} GB"
            )

    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def print_memory_state(label: str = "") -> None:
    """Print CPU and CUDA memory usage for debugging benchmark slowdowns."""
    import os

    try:
        import psutil
    except Exception as exc:
        print(f"psutil is not installed, so CPU RAM cannot be reported: {exc}")
        psutil = None

    gc.collect()
    print(f"\n[{label}]")

    if psutil is not None:
        process = psutil.Process(os.getpid())
        ram_gb = process.memory_info().rss / 1024**3
        print(f"CPU RAM used by Python process: {ram_gb:.3f} GB")

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        print(f"CUDA allocated:     {torch.cuda.memory_allocated() / 1024**3:.3f} GB")
        print(f"CUDA reserved:      {torch.cuda.memory_reserved() / 1024**3:.3f} GB")
        print(f"CUDA max allocated: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


def safe_import_faiss(verbose: bool = False):
    """Import FAISS safely and detect whether the imported build supports GPU."""
    try:
        import faiss  # type: ignore
    except Exception as exc:
        if verbose:
            print(f"{ts()} FAISS import failed: {exc}")
        return None, False, False

    has_gpu = hasattr(faiss, "StandardGpuResources") and hasattr(faiss, "GpuIndexFlatIP")
    has_cpu = hasattr(faiss, "IndexFlatIP")

    if verbose:
        print(f"{ts()} FAISS found: {getattr(faiss, '__file__', 'unknown')}")
        print(f"{ts()} FAISS version: {getattr(faiss, '__version__', 'unknown')}")
        print(f"{ts()} FAISS GPU support: {has_gpu}")
        print(f"{ts()} FAISS CPU IndexFlatIP support: {has_cpu}")

    return faiss, has_cpu, has_gpu


def diagnose_cosmap_environment() -> Dict[str, Any]:
    """Print and return basic environment information for CosMAP debugging."""
    import sys

    faiss, faiss_has_cpu, faiss_has_gpu = safe_import_faiss(verbose=False)
    info: Dict[str, Any] = {
        "python": sys.executable,
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_mps_available": bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ),
        "faiss_imported": faiss is not None,
        "faiss_file": getattr(faiss, "__file__", None) if faiss is not None else None,
        "faiss_version": getattr(faiss, "__version__", None) if faiss is not None else None,
        "faiss_has_cpu_indexflatip": bool(faiss_has_cpu),
        "faiss_has_gpu_standard_resources": bool(faiss_has_gpu),
        "cosmap_faiss_policy": "FAISS is used only when CUDA + FAISS-GPU are both available; otherwise torch GPU kNN is used when CUDA exists.",
    }
    for key, value in info.items():
        print(f"{key}: {value}")
    return info
