# @title Benchmark


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
                           davies_bouldin_score, calinski_harabasz_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import os
from datetime import datetime
import matplotlib

def evaluate_embeddings_multiple_runs(
    main_path,
    data_name,
    n_runs=5,
    embeddings_filename=None,
    timing_filename=None,
    output_folder_name=None
):
    """
    Evaluate embedding methods using multiple runs and compute average metrics.
    Now includes 5 clustering quality metrics.

    Parameters:
    - main_path: str, path to the folder containing embeddings and timing files
    - data_name: str, name of the dataset (e.g., 'pbmc', 'retina')
    - n_runs: int, number of runs to average over (default=5)
    - embeddings_filename: str, optional custom embeddings file name
    - timing_filename: str, optional custom timing file name
    - output_folder_name: str, optional custom output folder name

    Returns:
    - avg_metrics_dict: dict containing averaged metrics
    - out_dir: str, path to output directory
    """

    # Create output folder
    if output_folder_name is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = f"metrics_evaluation_{data_name}_{n_runs}runs_{stamp}"
    else:
        out_dir = output_folder_name

    os.makedirs(out_dir, exist_ok=True)
    print(f" Created output folder: {out_dir}")

    # Determine file paths
    if embeddings_filename is None:
        # Try to find the embeddings file automatically
        possible_files = [f for f in os.listdir(main_path) if f.startswith('all_embeddings_dict') and f.endswith('.npy')]
        if not possible_files:
            raise FileNotFoundError(f"No embeddings file found in {main_path}")
        embeddings_filename = possible_files[0]

    if timing_filename is None:
        # Try to find the timing file automatically
        possible_files = [f for f in os.listdir(main_path) if f.startswith('timing_dict') and f.endswith('.npy')]
        if not possible_files:
            print(" No timing file found, will skip timing information")
            timing_filename = None
        else:
            timing_filename = possible_files[0]

    embeddings_path = os.path.join(main_path, embeddings_filename)
    timing_path = os.path.join(main_path, timing_filename) if timing_filename else None

    print(f" Loading data from: {main_path}")
    print(f"   Embeddings: {embeddings_filename}")
    print(f"   Timing: {timing_filename if timing_filename else 'Not available'}")

    # Load embeddings and labels
    embeddings_dict = np.load(embeddings_path, allow_pickle=True).item()
    labels = embeddings_dict[f"labels_{data_name}"]

    # Load timing if available
    timing_dict = None
    if timing_path and os.path.exists(timing_path):
        timing_dict = np.load(timing_path, allow_pickle=True).item()
        print(f" Loaded timing information")

    # Handle categorical labels
    print("🏷️ Processing labels...")
    if not np.issubdtype(labels.dtype, np.number):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(labels)
        print(f"   Encoded {len(np.unique(labels))} categorical labels")
    else:
        y_encoded = labels
        print(f"   Using numerical labels")

    # Get embedding methods (exclude data keys)
    embedding_methods = [key for key in embeddings_dict.keys()
                        if key not in [f'X_{data_name}', f'labels_{data_name}', 'X', 'labels']]

    print(f" Found {len(embedding_methods)} embedding methods: {embedding_methods}")
    print(f" Running evaluation {n_runs} times...")

    # Initialize storage for all runs - NOW WITH 5 METRICS
    all_runs_metrics = {
        'silhouette': {method: [] for method in embedding_methods},      # Higher is better
        'ARI': {method: [] for method in embedding_methods},             # Higher is better
        'NMI': {method: [] for method in embedding_methods},             # Higher is better
        'davies_bouldin': {method: [] for method in embedding_methods},  # Lower is better
        'calinski_harabasz': {method: [] for method in embedding_methods} # Higher is better
    }

    # Run evaluation n times
    for run in range(n_runs):
        print(f"\n   Run {run + 1}/{n_runs}")

        for method in embedding_methods:
            print(f"      Evaluating {method}...")

            embedding = embeddings_dict[method]

            try:
                # Silhouette Score (deterministic - but we run multiple times for consistency)
                sil_score = silhouette_score(embedding, y_encoded)
                all_runs_metrics['silhouette'][method].append(sil_score)

                # For clustering-based metrics, use different random seeds for KMeans
                n_clusters = len(np.unique(y_encoded))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42+run, n_init=10)
                cluster_labels = kmeans.fit_predict(embedding)

                # Adjusted Rand Index (Higher is better)
                ari_score = adjusted_rand_score(y_encoded, cluster_labels)
                all_runs_metrics['ARI'][method].append(ari_score)

                # Normalized Mutual Information (Higher is better)
                nmi_score = normalized_mutual_info_score(y_encoded, cluster_labels)
                all_runs_metrics['NMI'][method].append(nmi_score)

                # Davies-Bouldin Index (Lower is better)
                db_score = davies_bouldin_score(embedding, cluster_labels)
                all_runs_metrics['davies_bouldin'][method].append(db_score)

                # Calinski-Harabasz Index (Higher is better)
                ch_score = calinski_harabasz_score(embedding, cluster_labels)
                all_runs_metrics['calinski_harabasz'][method].append(ch_score)

                print(f"         Sil: {sil_score:.3f}, ARI: {ari_score:.3f}, NMI: {nmi_score:.3f}")
                print(f"         DB: {db_score:.3f}, CH: {ch_score:.3f}")

            except Exception as e:
                print(f"          Error: {e}")
                all_runs_metrics['silhouette'][method].append(0)
                all_runs_metrics['ARI'][method].append(0)
                all_runs_metrics['NMI'][method].append(0)
                all_runs_metrics['davies_bouldin'][method].append(999)  # High value for "lower is better"
                all_runs_metrics['calinski_harabasz'][method].append(0)

    # Compute averages and standard deviations
    print(f"\n Computing averages across {n_runs} runs...")

    avg_metrics_dict = {
        'silhouette': {},
        'ARI': {},
        'NMI': {},
        'davies_bouldin': {},
        'calinski_harabasz': {}
    }

    std_metrics_dict = {
        'silhouette': {},
        'ARI': {},
        'NMI': {},
        'davies_bouldin': {},
        'calinski_harabasz': {}
    }

    for metric in ['silhouette', 'ARI', 'NMI', 'davies_bouldin', 'calinski_harabasz']:
        for method in embedding_methods:
            scores = all_runs_metrics[metric][method]
            avg_metrics_dict[metric][method] = np.mean(scores)
            std_metrics_dict[metric][method] = np.std(scores)

    # Save results
    avg_save_path = os.path.join(out_dir, f'avg_metrics_{data_name}_{n_runs}runs.npy')
    std_save_path = os.path.join(out_dir, f'std_metrics_{data_name}_{n_runs}runs.npy')
    all_runs_save_path = os.path.join(out_dir, f'all_runs_metrics_{data_name}_{n_runs}runs.npy')

    np.save(avg_save_path, avg_metrics_dict)
    np.save(std_save_path, std_metrics_dict)
    np.save(all_runs_save_path, all_runs_metrics)

    print(f" Saved average metrics: {avg_save_path}")
    print(f" Saved std metrics: {std_save_path}")
    print(f" Saved all runs: {all_runs_save_path}")

    # Define colors for each method
    color_map = {
        'cosmap_2d': 'red',
        'pacmap_2d': 'skyblue',
        'localmap_2d': 'orange',
        'tsne_2d': 'blue',
        'umap_2d': 'green',
        'hnne_2d': 'purple',
        'infonce_2d': 'brown',
        'negtsne_2d': 'pink',
        'ncvi_2d': 'gray',
        'trimap_2d': 'olive',
        'phate_2d': 'cyan',
        'pca_2d': 'yellow'
    }

    # Create plots with averages and error bars - NOW 5 PLOTS
    print(" Creating average metric plots with error bars...")

    # Define metrics with their properties
    metrics_info = [
        ('silhouette', 'Silhouette Score', 'higher'),
        ('ARI', 'Adjusted Rand Index', 'higher'),
        ('NMI', 'Normalized Mutual Info', 'higher'),
        ('davies_bouldin', 'Davies-Bouldin Index', 'lower'),
        ('calinski_harabasz', 'Calinski-Harabasz Index', 'higher')
    ]

    for metric, title, direction in metrics_info:
        print(f"   Creating plot for {title}...")

        fig, ax = plt.subplots(figsize=(16, 8))

        # Get average values and standard deviations
        methods = embedding_methods
        avg_values = [avg_metrics_dict[metric][method] for method in methods]
        std_values = [std_metrics_dict[metric][method] for method in methods]

        # Get colors
        bar_colors = [color_map.get(method, 'lightgray') for method in methods]

        # Find the best method based on direction
        if direction == 'higher':
            best_idx = np.argmax(avg_values)
            direction_text = "Higher is Better"
        else:  # direction == 'lower'
            best_idx = np.argmin(avg_values)
            direction_text = "Lower is Better"

        best_method = methods[best_idx]

        # Create bar plot with error bars
        bars = ax.bar(range(len(methods)), avg_values, yerr=std_values,
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1,
                     capsize=5, error_kw={'linewidth': 2})

        # Highlight best method
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(4)

        # Customize plot
        ax.set_title(f'{title} ({direction_text}) - Average of {n_runs} runs\n(Best: {best_method.replace("_2d", "").upper()})',
                    fontsize=20, fontweight='bold')
        ax.set_ylabel(f'{title} ± Std Dev', fontsize=16)

        # Set x-axis labels
        clean_labels = [method.replace('_2d', '').upper() for method in methods]
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(clean_labels, rotation=45, ha='right', fontsize=12)

        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        # Save plot
        plot_save_path = os.path.join(out_dir, f'{metric}_{data_name}_{n_runs}runs_avg.png')
        plt.savefig(plot_save_path, dpi=500, bbox_inches='tight', transparent=True)
        print(f" Saved: {plot_save_path}")
        plt.show()

    # Print comprehensive summary
    print(f"\n COMPREHENSIVE METRICS SUMMARY ({n_runs} runs)")
    print("="*80)

    for metric, title, direction in metrics_info:
        avg_values = [avg_metrics_dict[metric][method] for method in embedding_methods]
        std_values = [std_metrics_dict[metric][method] for method in embedding_methods]

        if direction == 'higher':
            best_idx = np.argmax(avg_values)
            direction_symbol = ""
        else:
            best_idx = np.argmin(avg_values)
            direction_symbol = "📉"

        best_method = embedding_methods[best_idx]
        best_avg = avg_values[best_idx]
        best_std = std_values[best_idx]

        print(f"\n {title.upper()} {direction_symbol} ({direction} is better)")
        print(f"   Best: {best_method.replace('_2d', '').upper()} ({best_avg:.4f} ± {best_std:.4f})")

        # Show all scores
        for method, avg_score, std_score in zip(embedding_methods, avg_values, std_values):
            clean_name = method.replace('_2d', '').upper()
            star = " ⭐" if method == best_method else ""
            print(f"      {clean_name:15} | {avg_score:8.4f} ± {std_score:.4f}{star}")

    print(f"\n Completed comprehensive evaluation with {n_runs} runs!")
    print(f" Evaluated 5 clustering quality metrics:")
    print(f"   • Silhouette Score (higher=better)")
    print(f"   • Adjusted Rand Index (higher=better)")
    print(f"   • Normalized Mutual Info (higher=better)")
    print(f"   • Davies-Bouldin Index (lower=better)")
    print(f"   • Calinski-Harabasz Index (higher=better)")
    print(f" Results saved in: {out_dir}")

    return avg_metrics_dict, out_dir



import time
import numpy as np
from tqdm import tqdm
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# =================================================================
# EXAMPLE USAGE FUNCTIONS
# =================================================================

def run_standard_benchmark(data_name, random_state=2025):
    """Run standard benchmark with default settings"""
    return run_dimensionality_reduction_benchmark(
        data_name=data_name,
        random_state=random_state
    )

def run_custom_benchmark(data_name, methods_list, custom_cosmap_params=None):
    """Run benchmark with custom method selection and CosMAP parameters"""
    return run_dimensionality_reduction_benchmark(
        data_name=data_name,
        methods_to_run=methods_list,
        cosmap_params=custom_cosmap_params,
        save_individual_files=True
    )

def run_quick_benchmark(data_name, random_state=2025):
    """Run quick benchmark with just the most common methods"""
    quick_methods = ['cosmap_2d', 'umap_2d', 'tsne_2d', 'pacmap_2d', 'pca_2d']
    return run_dimensionality_reduction_benchmark(
        data_name=data_name,
        random_state=random_state,
        methods_to_run=quick_methods
    )

# =================================================================
# MAIN EXECUTION EXAMPLES
# =================================================================

# if __name__ == "__main__":
#     # Example 1: Standard benchmark
#     print("Example 1: Standard benchmark")
#     embeddings, timings, folder = run_standard_benchmark("cortex", random_state=2025)

#     # Example 2: Custom benchmark
#     print("\nExample 2: Custom benchmark")
#     custom_methods = ['cosmap_2d', 'umap_2d', 'tsne_2d']
#     custom_cosmap = {'n_neighbors': 10, 'temperature': 0.3}
#     embeddings, timings, folder = run_custom_benchmark(
#         "cortex",
#         custom_methods,
#         custom_cosmap
#     )

#     # Example 3: Quick benchmark
#     print("\nExample 3: Quick benchmark")
#     embeddings, timings, folder = run_quick_benchmark("cortex")



import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def get_colors_for_labels(unique_labels, custom_color_map=None):
    """
    Color logic:
    - If custom_color_map is provided: use it, regardless of the number of labels
    - Else if ≤10 labels: use tab10
    - Else if ≤20 labels: use tab20
    - Else: raise an error because custom colors are required
    """
    n_labels = len(unique_labels)

    # --------------------------------------------------
    # 1. Custom colors have priority
    # --------------------------------------------------
    if custom_color_map is not None:
        matching_labels = set(unique_labels) & set(custom_color_map.keys())

        if len(matching_labels) == 0:
            raise ValueError(
                "No labels match between data and custom_color_map!\n"
                f"Data labels: {list(unique_labels)}\n"
                f"Custom colors: {list(custom_color_map.keys())}"
            )

        print(f"Using CUSTOM colors for {n_labels} labels...")

        colors = []

        for label in unique_labels:
            if label in custom_color_map:
                custom_color = custom_color_map[label]

                try:
                    rgba_color = mcolors.to_rgba(custom_color)
                    colors.append(rgba_color)
                    # print(f"    {label} -> {custom_color}")

                except ValueError:
                    print(f"    Invalid color '{custom_color}' for {label}, using gray")
                    colors.append(mcolors.to_rgba("gray"))

            else:
                print(f"    No color specified for {label}, using gray")
                colors.append(mcolors.to_rgba("gray"))

        return colors

    # --------------------------------------------------
    #  Default colors when no custom map is provided
    # --------------------------------------------------
    if n_labels <= 10:
        # print(f"Using tab10 colors for {n_labels} labels...")
        return plt.cm.tab10(np.linspace(0, 1, n_labels))

    if n_labels <= 20:
        # print(f"Using tab20 colors for {n_labels} labels...")
        return plt.cm.tab20(np.linspace(0, 1, n_labels))

    # --------------------------------------------------
    # Too many labels without custom colors
    # --------------------------------------------------
    raise ValueError(
        f"Dataset has {n_labels} labels (>20). "
        "You MUST provide custom_color_map!"
    )

def plot_embeddings_comparison_v0(
    embeddings_dict,
    data_name,
    random_state=2025,
    custom_colors=None,
    output_dir=None,
    dpi=700,
    figure_size=(24, 18),
    point_size=5,
    save_plot=True,
    show_plot=True
):
    """
    Create a 3x4 comparison plot of embedding methods.

    Color logic:
    - If custom_color_map is provided: use it, regardless of the number of labels
    - Else if ≤10 labels: use tab10
    - Else if ≤20 labels: use tab20
    - Else: raise an error because custom colors are required

    Parameters:
    -----------
    embeddings_dict : dict
        Dictionary containing embeddings and labels
    data_name : str
        Name of the dataset
    custom_colors : dict, REQUIRED if >10 labels
        Dictionary mapping label names to colors
    ... other parameters same as before
    """

    # Extract labels
    y = embeddings_dict[f"labels_{data_name}"]
    print(f"Shape of y={y.shape}", "=="*10)
    # Handle categorical labels
    print(" Processing labels...")
    if not np.issubdtype(y.dtype, np.number):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        unique_labels = label_encoder.classes_
        print(f"   Found {len(unique_labels)} categorical labels: {list(unique_labels)}")
    else:
        y_encoded = y
        unique_labels = np.unique(y)
        print(f"   Found {len(unique_labels)} numerical labels: {list(unique_labels)}")

    # Get colors - SIMPLE LOGIC
    distinct_colors = get_colors_for_labels(unique_labels, custom_colors)

    # Create output directory
    if output_dir is None:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"embeddings_plot_{data_name}_seed_{random_state}_{stamp}"

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        print(f" Created output folder: {output_dir}")

    # Setup the 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=figure_size)

    # Define method order
    method_order = [
        'cosmap_2d',    'pacmap_2d',    'localmap_2d',  'tsne_2d',
        'umap_2d',      'hnne_2d',      'infonce_2d',   'negtsne_2d',
        'ncvi_2d',      'trimap_2d',    'phate_2d'
    ]

    print(f"\n Creating 3x4 comparison plot...")

    # Plot each method
    for idx, method in enumerate(method_order):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        if method in embeddings_dict:
            embedding_data = embeddings_dict[method]
            print(f"   Plotting {method} at position [{row},{col}]")

            # Plot each class
            print(f"Shape of embedding_data={embedding_data.shape}", "=="*10,'\n')
            for i, label in enumerate(unique_labels):
                label_idx = y == label

                if np.any(label_idx):
                    ax.scatter(
                        embedding_data[label_idx, 0],
                        embedding_data[label_idx, 1],
                        label=label,
                        alpha=1,
                        color=distinct_colors[i],
                        s=point_size
                    )

            clean_title = method.replace('_2d', '').upper()
            ax.set_title(clean_title, fontsize=10)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")

        else:
            print(f"     Method {method} not found")
            ax.text(0.5, 0.5, f'{method}\nNot Available',
                    ha='center', va='center', fontsize=20, transform=ax.transAxes)
            ax.set_title(method.replace('_2d', '').upper(), fontsize=30)

    # CREATE LEGEND
    print(" Creating legend...")
    legend_ax = axes[2, 3]
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    n_labels = len(unique_labels)
    max_cols = 2 if n_labels > 10 else 1
    n_per_col = int(np.ceil(n_labels / max_cols))

    # Legend title
    color_type = "tab10" if n_labels <= 10 else "Custom"
    # legend_ax.text(0.5, 0.98, f'Class Labels ({n_labels})\n{color_type} Colors',
    #                ha='center', va='top', fontsize=12, fontweight='bold',
    #                transform=legend_ax.transAxes)

    legend_ax.text(0.5, 0.98, f' Labels ',
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   transform=legend_ax.transAxes)

    # Plot legend items
    col_width = 0.4
    row_height = 0.85 / n_per_col

    for i, label in enumerate(unique_labels):
        col = i // n_per_col
        row = i % n_per_col

        x = 0.05 + col * col_width
        y = 0.9 - row * row_height

        # Color circle
        circle = plt.Circle((x, y), 0.02, color=distinct_colors[i],
                           transform=legend_ax.transAxes, edgecolor='black', linewidth=0.5)
        legend_ax.add_patch(circle)

        # Label text
        legend_ax.text(x + 0.03, y, str(label), transform=legend_ax.transAxes,
                      fontsize=8, va='center', ha='left')

    plt.tight_layout()

    # Save plot
    output_path = None
    if save_plot:
        output_path = os.path.join(output_dir, f"embeddings_comparison_{data_name}_seed_{random_state}.png")
        plt.savefig(output_path, dpi=dpi, transparent=True, bbox_inches='tight')
        print(f" Saved plot: {output_path}")

    if show_plot:
        plt.show()

    # Print summary
    print(f"\n COLOR SUMMARY:")
    print("="*50)
    if n_labels <= 10:
        print(f" Used tab10 colors for {n_labels} labels")
    else:
        print(f" Used custom colors for {n_labels} labels")
        for i, label in enumerate(unique_labels):
            if custom_colors and label in custom_colors:
                print(f"   {label} -> {custom_colors[label]}")

    return fig, output_path

# =================================================================
# PREDEFINED COLORS FOR DATASETS WITH >10 LABELS
# =================================================================

PBMC_COLORS = {
    "B cells": "#1f77b4",
    "CD14+ Monocytes": "#ff7f0e",
    "CD4 T cells": "#2ca02c",
    "CD8 T cells": "#d62728",
    "Dendritic Cells": "#9467bd",
    "FCGR3A+ Monocytes": "#8c564b",
    "Megakaryocytes": "#e377c2",
    "NK cells": "#17becf",
    "Other": "#7f7f7f"
}

# RETINA_COLORS = {
#     "BC1A": "red", "BC1B": "black", "BC2": "blue", "BC3A": "yellow",
#     "BC3B": "magenta", "BC4": "cyan", "BC5A": "orange", "BC5B": "purple",
#     "BC5C": "pink", "BC5D": "brown", "BC6": "gray", "BC7": "navy",
#     "BC8_9": "lightgreen", "MG": "maroon", "RBC": "green"
# }
RETINA_COLORS = {
    "BC1A": "red", "BC1B": "black", "BC2": "blue", "BC3A": "#D191EB",
    "BC3B": "magenta", "BC4": "cyan", "BC5A": "green", "BC5B": "purple",
    "BC5C": "pink", "BC5D": "brown", "BC6": "gray", "BC7": "navy",
    "BC8_9": "lightgreen", "MG": "maroon", "RBC": "orange"
}

# RETINA_COLORS = {
#     "BC1A": "black",
#     "BC1B": "black",
#     "BC2": "black",
#     "BC3A": "black",
#     "BC3B": "black",
#     "BC4": "black",
#     "BC5A": "black",
#     "BC5B": "black",
#     "BC5C": "black",
#     "BC5D": "black",
#     "BC6": "black",
#     "BC7": "black",
#     "BC8_9": "black",
#     "MG": "black",
#     "RBC": "black",
# }
# =================================================================
# EXAMPLE USAGE
# =================================================================

# KINSHIP_COLORS = {
#     # ─── Greater Montréal cluster : blue / turquoise family ───
#     "Ile-de-Montréal": "#1565c0",              # deep blue
#     "Rive Nord-Ouest de Montréal": "#1e88e5", # medium blue
#     "Rive Sud de Montréal": "#26a69a",         # teal
#     "Outaouais": "#00897b",                    # dark teal
#     "Lanaudière": "#42a5f5",                   # light blue
#     "Laurentides": "#5c6bc0",                  # blue-indigo
#     "Richelieu": "#7e57c2",                    # violet-blue

#     # ─── Québec / Capitale-Nationale / nearby corridor : purple family ───
#     "Québec": "#4527a0",                       # strong purple
#     "Région de Québec": "#5e35b1",             # royal purple
#     "Agglomération de Québec": "#673ab7",      # medium purple
#     "Côte-de-Beaupré": "#7e57c2",              # lighter purple
#     "Portneuf": "#9575cd",                     # soft purple
#     "Lévis-Lotbinière": "#8e24aa",             # purple-magenta
#     "Côte-du-Sud": "#ab47bc",                  # light violet
#     "Beauce": "#ba68c8",                       # violet-pink
#     "Estrie": "#8e24aa",                       # same broad southeast family

#     # ─── Saguenay–Lac-Saint-Jean / Charlevoix : green family ───
#     "Saguenay-Lac-St-Jean": "#1b5e20",         # dark green
#     "Charlevoix": "#2e7d32",                   # green
#     "Mauricie": "#388e3c",                     # medium green
#     "Bois-Francs": "#66bb6a",                  # light green

#     # ─── Lower St. Lawrence / North Shore / Gaspésie : warm coastal family ───
#     "Bas-Saint-Laurent": "#f9a825",            # amber
#     "Côte-Nord": "#fb8c00",                    # orange
#     "Gaspésie": "#f4511e",                     # deep orange
#     "Iles-de-la-Madeleine": "#ec407a",         # pink

#     # ─── Abitibi / Témiscamingue / Nord-du-Québec : earthy remote family ───
#     "Abitibi": "#8d6e63",                      # earthy brown
#     "Témiscamingue": "#6d4c41",                # dark brown
#     "Nord du Québec": "#78909c",               # blue-grey

#     # ─── Fallback ───
#     "Unknown": "#d9d9d9",
# }

KINSHIP_COLORS = {
    # ─── Greater Montréal cluster : orange family ───
    "Ile-de-Montréal": "#e65100",
    "Rive Nord-Ouest de Montréal": "#ef6c00",
    "Rive Sud de Montréal": "#f57c00",
    "Outaouais": "#fb8c00",
    "Lanaudière": "#ff9800",
    "Laurentides": "#ffa726",
    "Richelieu": "#ffb74d",

    # ─── Québec / Capitale-Nationale / nearby corridor : purple family ───
    "Québec": "#4527a0",
    "Région de Québec": "#5e35b1",
    "Agglomération de Québec": "#673ab7",
    "Côte-de-Beaupré": "#7e57c2",
    "Portneuf": "#9575cd",
    "Lévis-Lotbinière": "#8e24aa",
    "Côte-du-Sud": "#ab47bc",
    "Beauce": "#ba68c8",
    "Estrie": "#8e24aa",

    # ─── Saguenay–Lac-Saint-Jean / Charlevoix : green family ───
    "Saguenay-Lac-St-Jean": "#1b5e20",
    "Charlevoix": "#2e7d32",
    "Mauricie": "#388e3c",
    "Bois-Francs": "#66bb6a",

    # ─── Lower St. Lawrence / North Shore / Gaspésie : blue family ───
    "Bas-Saint-Laurent": "#0d47a1",
    "Côte-Nord": "#1565c0",
    "Gaspésie": "#1e88e5",
    "Iles-de-la-Madeleine": "#64b5f6",

    # ─── Abitibi / Témiscamingue / Nord-du-Québec : small remote group ───
    "Abitibi": "#795548",
    "Témiscamingue": "#a1887f",
    "Nord du Québec": "#455a64",

    # ─── Fallback ───
    "Unknown": "#d9d9d9",
}
# If we need black and white for some reason, we uncomment the following block and use it as the custom color map:

# KINSHIP_COLORS = {
#     # ─── Greater Montréal cluster ───
#     "Ile-de-Montréal": "#000000",
#     "Rive Nord-Ouest de Montréal": "#000000",
#     "Rive Sud de Montréal": "#000000",
#     "Outaouais": "#000000",
#     "Lanaudière": "#000000",
#     "Laurentides": "#000000",
#     "Richelieu": "#000000",

#     # ─── Québec / Capitale-Nationale / nearby corridor ───
#     "Québec": "#000000",
#     "Région de Québec": "#000000",
#     "Agglomération de Québec": "#000000",
#     "Côte-de-Beaupré": "#000000",
#     "Portneuf": "#000000",
#     "Lévis-Lotbinière": "#000000",
#     "Côte-du-Sud": "#000000",
#     "Beauce": "#000000",
#     "Estrie": "#000000",

#     # ─── Saguenay–Lac-Saint-Jean / Charlevoix ───
#     "Saguenay-Lac-St-Jean": "#000000",
#     "Charlevoix": "#000000",
#     "Mauricie": "#000000",
#     "Bois-Francs": "#000000",

#     # ─── Lower St. Lawrence / North Shore / Gaspésie ───
#     "Bas-Saint-Laurent": "#000000",
#     "Côte-Nord": "#000000",
#     "Gaspésie": "#000000",
#     "Iles-de-la-Madeleine": "#000000",

#     # ─── Abitibi / Témiscamingue / Nord-du-Québec ───
#     "Abitibi": "#000000",
#     "Témiscamingue": "#000000",
#     "Nord du Québec": "#000000",

#     # ─── Fallback ───
#     "Unknown": "#000000",
# }

from typing import Dict

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
    "infonce_2d": "InfoCE-t-SNE",
    "negtsne_2d": "Neg-t-SNE",
    "ncvi_2d": "NCVis",
}


def plot_embeddings_comparison(
    embeddings_dict,
    data_name,
    random_state=2025,
    custom_colors=None,
    exclude_labels=None,
    labels_name="Labels",
    label_names=None,
    output_dir=None,
    dpi=700,
    figure_size=(24, 18),
    point_size=5,
    save_plot=True,
    show_plot=True,
    method_display_names=None,
):
    """
    Create a 3x4 comparison plot of embedding methods.

    New options:
    - exclude_labels: labels to remove from both the scatter plots and the legend.
      Example: exclude_labels=["Unknown"].
    - labels_name: legend title.
      Example: "Region" for kinship or "Cell type" for scRNA-seq.
    - label_names: optional dictionary mapping label values to display names.
      Example: {0: "T-shirt/top", 1: "Trouser", ...} for F-MNIST.
    - method_display_names: optional dictionary mapping method keys to display names.
    """

    if exclude_labels is None:
        exclude_labels = []
    exclude_labels = set(exclude_labels)

    if method_display_names is None:
        method_display_names = METHOD_DISPLAY_NAMES

    # Extract labels
    y = np.asarray(embeddings_dict[f"labels_{data_name}"])
    print(f"Shape of y={y.shape}", "==" * 10)

    # ------------------------------------------------------------
    # Filter labels BEFORE computing unique labels and colors.
    # This ensures excluded labels are not plotted and not shown
    # in the legend.
    # ------------------------------------------------------------
    keep_mask = np.ones(y.shape[0], dtype=bool)

    if exclude_labels:
        keep_mask = ~np.isin(
            y.astype(str),
            [str(label) for label in exclude_labels]
        )
        n_removed = int(np.sum(~keep_mask))
        # print(f" Excluding labels {list(exclude_labels)}: removed {n_removed} points")

    y_plot = y[keep_mask]

    if y_plot.size == 0:
        raise ValueError(
            "No points left to plot after applying exclude_labels. "
            f"exclude_labels={list(exclude_labels)}"
        )

    # Handle labels after filtering
    # print(" Processing labels...")
    if not np.issubdtype(y_plot.dtype, np.number):
        label_encoder = LabelEncoder()
        label_encoder.fit(y_plot)
        unique_labels = label_encoder.classes_
        # print(
        #     f"   Found {len(unique_labels)} categorical labels after filtering: "
        #     f"{list(unique_labels)}"
        # )
    else:
        unique_labels = np.unique(y_plot)
        # print(
        #     f"   Found {len(unique_labels)} numerical labels after filtering: "
        #     f"{list(unique_labels)}"
        # )

    # Get colors only for visible labels
    distinct_colors = get_colors_for_labels(unique_labels, custom_colors)

    # Create output directory
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # output_dir = f"embeddings_plot_{data_name}_seed_{random_state}_{stamp}"
        output_dir = f"embeddings_plot_{data_name}_seed_{random_state}"

    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        print(f" Created output folder: {output_dir}")

    # Setup 3x4 grid
    fig, axes = plt.subplots(3, 4, figsize=figure_size)

    method_order = [
        "cosmap_2d", "pacmap_2d", "localmap_2d", "tsne_2d",
        "umap_2d", "hnne_2d", "infonce_2d", "negtsne_2d",
        "ncvi_2d", "trimap_2d", "phate_2d"
    ]

    # print("\n Creating 3x4 comparison plot...")

    for idx, method in enumerate(method_order):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        display_name = method_display_names.get(
            method,
            method.replace("_2d", "").upper()
        )

        if method in embeddings_dict:
            embedding_data = np.asarray(embeddings_dict[method])
            embedding_plot = embedding_data[keep_mask]

            # print(f"   Plotting {display_name} at position [{row},{col}]")
            # print(
            #     f"Shape of embedding_data after filtering={embedding_plot.shape}",
            #     "==" * 10,
            #     "\n"
            # )

            for i, label in enumerate(unique_labels):
                label_idx = y_plot == label

                if np.any(label_idx):
                    ax.scatter(
                        embedding_plot[label_idx, 0],
                        embedding_plot[label_idx, 1],
                        label=str(label),
                        alpha=1,
                        color=distinct_colors[i],
                        s=point_size
                    )

            ax.set_title(display_name, fontsize=10)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")

        else:
            print(f"     Method {method} not found")
            ax.text(
                0.5,
                0.5,
                f"{display_name}\nNot Available",
                ha="center",
                va="center",
                fontsize=20,
                transform=ax.transAxes
            )
            ax.set_title(display_name, fontsize=10)

    # Legend in the empty 12th panel
    # print(" Creating legend...")
    legend_ax = axes[2, 3]
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis("off")

    n_labels = len(unique_labels)
    max_cols = 2 if n_labels > 10 else 1
    n_per_col = int(np.ceil(n_labels / max_cols))

    legend_ax.text(
        0.5,
        0.98,
        labels_name,
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        transform=legend_ax.transAxes
    )

    col_width = 0.46 if max_cols == 2 else 0.85
    row_height = 0.85 / max(n_per_col, 1)

    for i, label in enumerate(unique_labels):
        col = i // n_per_col
        row = i % n_per_col

        x = 0.05 + col * col_width
        y_pos = 0.9 - row * row_height

        circle = plt.Circle(
            (x, y_pos),
            0.02,
            color=distinct_colors[i],
            transform=legend_ax.transAxes,
            edgecolor="black",
            linewidth=0.5
        )
        legend_ax.add_patch(circle)

        # Use label_names mapping if provided, otherwise use label as-is
        display_label = label
        if label_names is not None and label in label_names:
            display_label = label_names[label]

        legend_ax.text(
            x + 0.03,
            y_pos,
            str(display_label),
            transform=legend_ax.transAxes,
            fontsize=8,
            va="center",
            ha="left"
        )

    plt.tight_layout()

    output_path = None
    if save_plot:
        output_path = os.path.join(
            output_dir,
            f"embeddings_comparison_{data_name}_seed_{random_state}.png"
        )
        plt.savefig(output_path, dpi=dpi, transparent=False, bbox_inches="tight")
        print(f" Saved plot: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    print("\n COLOR SUMMARY:")
    print("=" * 50)
    print(f" Legend title: {labels_name}")

    if exclude_labels:
        print(f" Excluded labels: {list(exclude_labels)}")

    # if custom_colors is None and n_labels <= 10:
    #     print(f" Used tab10 colors for {n_labels} labels")
    # elif custom_colors is None and n_labels <= 20:
    #     print(f" Used tab20 colors for {n_labels} labels")
    # else:
    #     print(f" Used custom colors for {n_labels} labels")
    #     for label in unique_labels:
    #         if custom_colors and label in custom_colors:
    #             print(f"   {label} -> {custom_colors[label]}")

    return fig, output_path
if __name__ == "__main__":
    # Example with ≤10 labels (automatic tab10)
    # embeddings_dict = np.load('embeddings_small_dataset.npy', allow_pickle=True).item()
    # fig, path = plot_embeddings_comparison(embeddings_dict, "small_dataset")

    # Example with >10 labels (MUST provide custom colors)
    # embeddings_dict = np.load('embeddings_retina.npy', allow_pickle=True).item()
    # fig, path = plot_embeddings_comparison(embeddings_dict, "retina", custom_colors=RETINA_COLORS)

    pass

