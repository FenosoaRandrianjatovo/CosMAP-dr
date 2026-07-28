#!/usr/bin/env python3
"""
Author: Fenosoa Randrianjatovo 
Run one dataset from the CosMAP benchmark as a Slurm array task.

"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
from pathlib import Path

EXPERIMENT_DIR = Path(
    "/home/fenosoa/links/projects/def-amadou/fenosoa/"
    "Notebook_cosmap/CosMAP-dr/experiments"
)
DATA_PATH = Path(
    "/home/fenosoa/links/projects/def-amadou/fenosoa/data_benchmark"
)

DATA_NAMES = [
    "retina",
    "cortex",
    "pbmc",
    "paul15",
    "fmnist",
    "coil_20",
    "20NG",
    "mnist",
    "USPS",
    "heart_cell_atlas",
    "kinship_rrq",
    "kinship_cartagene",
]

METHODS = [
    "cosmap_2d",
    "pacmap_2d",
    "umap_2d",
    "localmap_2d",
    "tsne_2d",
    "trimap_2d",
    "phate_2d",
    "pca_2d",
    "hnne_2d",
    "infonce_2d",
    "negtsne_2d",
    "ncvi_2d",
]

KINSHIP_DATASETS = {"kinship_rrq", "kinship_cartagene"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one dataset from the dimensionality-reduction benchmark."
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help=f"Zero-based dataset index, from 0 to {len(DATA_NAMES) - 1}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("BENCHMARK_SEED", "42")),
        help="Random seed. Defaults to BENCHMARK_SEED or 42.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when a successful-completion marker already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not 0 <= args.index < len(DATA_NAMES):
        raise ValueError(
            f"Invalid index {args.index}; expected 0 to {len(DATA_NAMES) - 1}."
        )

    if not EXPERIMENT_DIR.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {EXPERIMENT_DIR}")
    if not DATA_PATH.is_dir():
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

    os.chdir(EXPERIMENT_DIR)
    sys.path.insert(0, str(EXPERIMENT_DIR))

    # Import only after changing to the directory containing benchmark_pipeline.py.
    from benchmark_pipeline import run_dimensionality_reduction_benchmark

    data_name = DATA_NAMES[args.index]
    output_name = f"comparison_with_seed_{args.seed}"
    output_dir = EXPERIMENT_DIR / output_name
    done_dir = output_dir / ".done"
    done_marker = done_dir / f"{data_name}_seed_{args.seed}.done"

    if done_marker.exists() and not args.force:
        print(f"[SKIP] Successful completion marker already exists: {done_marker}")
        print("[SKIP] Submit with BENCHMARK_FORCE=1 to rerun this dataset.")
        return 0

    done_dir.mkdir(parents=True, exist_ok=True)

    cosmap_params = {
        "n_components": 2,
        "n_neighbors": 15,
        "temperature": 0.5,
        "use_gpu": 1,
        "random_state": args.seed,
        "optimizer_backend": "torch_manual",
        "faiss_backend": "none",
        "refinement": True,
        "refinement_dim": 30,
        "refinement_n_neighbors": 30,
        "verbose": True,
    }

    # Only the two kinship datasets use Euclidean distance.
    # For every other dataset, omit "metric" so CosMAP uses its default.
    if data_name in KINSHIP_DATASETS:
        cosmap_params["metric"] = "euclidean"

    started = time.time()
    print("=" * 80)
    print(f"Host:              {socket.gethostname()}")
    print(f"Slurm job ID:      {os.environ.get('SLURM_JOB_ID', 'not-set')}")
    print(f"Array task ID:     {os.environ.get('SLURM_ARRAY_TASK_ID', 'not-set')}")
    print(f"Dataset index:     {args.index}")
    print(f"Dataset:           {data_name}")
    print(f"Seed:              {args.seed}")
    print(f"CosMAP metric:     {cosmap_params.get('metric', '<CosMAP default>')}")
    print(f"CPU count:         {os.environ.get('SLURM_CPUS_PER_TASK', 'not-set')}")
    print(f"CUDA devices:      {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    print(f"Output directory:  {output_dir}")
    print("=" * 80, flush=True)

    try:
        embeddings_dict, timing_dict, returned_output_folder = (
            run_dimensionality_reduction_benchmark(
                data_name=data_name,
                data_path=str(DATA_PATH),
                methods_to_run=METHODS,
                random_state=args.seed,
                output_dir=output_name,
                save_individual_files=True,
                cosmap_params=cosmap_params,
            )
        )
    except Exception:
        print(f"[FAILED] Dataset: {data_name}", file=sys.stderr)
        traceback.print_exc()
        return 1

    elapsed = time.time() - started
    done_marker.write_text(
        "\n".join(
            [
                f"dataset={data_name}",
                f"seed={args.seed}",
                f"elapsed_seconds={elapsed:.6f}",
                f"slurm_job_id={os.environ.get('SLURM_JOB_ID', '')}",
                f"slurm_array_task_id={os.environ.get('SLURM_ARRAY_TASK_ID', '')}",
                f"output_folder={returned_output_folder}",
                f"embedding_keys={sorted(map(str, embeddings_dict.keys()))}",
                f"timing_keys={sorted(map(str, timing_dict.keys()))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[DONE] Dataset: {data_name}")
    print(f"[DONE] Elapsed: {elapsed / 60:.2f} minutes")
    print(f"[DONE] Marker:  {done_marker}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
