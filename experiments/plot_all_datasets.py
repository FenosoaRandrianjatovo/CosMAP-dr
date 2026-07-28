"""
Author: Fenosoa Randrianjatovo 



This file is designed to be used directly in a notebook or as a Python module.


Script to plot all embedding comparison visualizations for all datasets.
Uses the plot_embeddings_comparison function from visualition.py
"""

import numpy as np
import os
import sys
from datetime import datetime
import matplotlib

# Add the current directory to the path to import visualition
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualisation import plot_embeddings_comparison, PBMC_COLORS, RETINA_COLORS, KINSHIP_COLORS

# F-MNIST class mapping
FMNIST_COLORS = {
    0: "#1f77b4",    # T-shirt/top (blue)
    1: "#ff7f0e",    # Trouser (orange)
    2: "#2ca02c",    # Pullover (green)
    3: "#d62728",    # Dress (red)
    4: "#9467bd",    # Coat (purple)
    5: "#8c564b",    # Sandal (brown)
    6: "#e377c2",    # Shirt (pink)
    7: "#7f7f7f",    # Sneaker (gray)
    8: "#bcbd22",    # Bag (olive)
    9: "#17becf",    # Ankle boot (cyan)
}

# Map numeric labels to class names for display in legend
FMNIST_LABEL_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

# Dataset configurations
DATASETS = [
    {
        'filename': 'all_embeddings_dict_23_mnist.npy',
        'data_name': 'mnist',
        'custom_colors': None,
        'point_size': 1,
        'labels_name': "Class",
    },
    {
        'filename': 'all_embeddings_dict_23_20NG.npy',
        'data_name': '20NG',
        'custom_colors': None,
        'point_size': 2,
        'labels_name': "Class",
    },
    {
        'filename': 'all_embeddings_dict_23_coil_20.npy',
        'data_name': 'coil_20',
        'custom_colors': None,
        'point_size': 2,
        'labels_name': "Class",
    },
    {
        'filename': 'all_embeddings_dict_23_fmnist.npy',
        'data_name': 'fmnist',
        'custom_colors': FMNIST_COLORS,
        'point_size': 1,
        'labels_name': "Class",
        'label_names': FMNIST_LABEL_NAMES,
    },
    {
        'filename': 'all_embeddings_dict_23_heart_cell_atlas.npy',
        'data_name': 'heart_cell_atlas',
        'custom_colors': None,
        'point_size': 2,
        'labels_name': "Cell Type",
    },
    {
        'filename': '__comparison_kinship_cartagene_seed_42/all_embeddings_dict_42_kinship_cartagene.npy',
        'data_name': 'kinship_cartagene',
        'custom_colors': KINSHIP_COLORS,
        'point_size': 2,
        'labels_name': "Region",
        'exclude_labels': ["Unknown"],
    },
    {
        'filename': 'all_embeddings_dict_23_pbmc.npy',
        'data_name': 'pbmc',
        'custom_colors': PBMC_COLORS,
        'point_size': 3,
        'labels_name': "Cell Type",
    },
        {
        'filename': 'all_embeddings_dict_23_USPS.npy',
        'data_name': 'USPS',
        'custom_colors': None,
        'point_size': 2,
        'labels_name': "Class",
    },

     {
        'filename': '__comparison_cortex_seed_42/all_embeddings_dict_42_cortex.npy',
        'data_name': 'cortex',
        'custom_colors': None,
        'point_size': 5,
        'labels_name': "Cell Type",
    },
    {
        'filename': 'all_embeddings_dict_23_retina.npy',
        'data_name': 'retina',
        'custom_colors': RETINA_COLORS,
        'point_size': 3,
        'labels_name': "Cell Type",
    },
    {
        'filename': 'embeddings_dict_cortex.npy',
        'data_name': 'cortex',
        'custom_colors': None,
        'point_size': 10,
        'labels_name': "Cell Type",
    },
]

def plot_all_datasets(base_path=None, random_state=2025, dpi=700, figure_size=(24, 18), show_plots=True):
    """
    Plot all datasets using plot_embeddings_comparison.
    
    Parameters:
    -----------
    base_path : str, optional
        Path to the folder containing embedding files. If None, uses current directory.
    random_state : int
        Random state for reproducibility (default: 2025)
    dpi : int
        DPI for saved figures (default: 700)
    figure_size : tuple
        Figure size (width, height) in inches (default: (24, 18))
    show_plots : bool
        Whether to display plots (default: True)
    """
    
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"📊 Starting batch plotting of all datasets")
    print(f"   Base path: {base_path}")
    print(f"   Random state: {random_state}")
    print(f"   DPI: {dpi}")
    print(f"   Figure size: {figure_size}")
    print("=" * 80)
    
    total_datasets = len(DATASETS)
    successful_plots = 0
    failed_plots = []
    
    for idx, dataset_config in enumerate(DATASETS, 1):
        filename = dataset_config['filename']
        data_name = dataset_config['data_name']
        custom_colors = dataset_config['custom_colors']
        point_size = dataset_config['point_size']
        labels_name = dataset_config.get('labels_name', 'Labels')
        exclude_labels = dataset_config.get('exclude_labels', None)
        label_names = dataset_config.get('label_names', None)
        
        file_path = os.path.join(base_path, filename)
        
        print(f"\n[{idx}/{total_datasets}] Processing: {data_name.upper()}")
        print(f"   File: {filename}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"   ❌ ERROR: File not found - {file_path}")
            failed_plots.append((data_name, "File not found"))
            continue
        
        try:
            print(f"   Loading embeddings...")
            embeddings_dict = np.load(file_path, allow_pickle=True).item()
            
            print(f"   Creating comparison plot...")
            fig, output_path = plot_embeddings_comparison(
                embeddings_dict,
                data_name=data_name,
                random_state=random_state,
                custom_colors=custom_colors,
                exclude_labels=exclude_labels,
                labels_name=labels_name,
                label_names=label_names,
                output_dir=None,
                dpi=dpi,
                figure_size=figure_size,
                point_size=point_size,
                save_plot=True,
                show_plot=show_plots
            )
            
            print(f"   ✅ Successfully plotted and saved!")
            successful_plots += 1
            
        except KeyError as e:
            error_msg = f"KeyError: Missing key {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_plots.append((data_name, error_msg))
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"   ❌ ERROR: {error_msg}")
            failed_plots.append((data_name, error_msg))
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 BATCH PLOTTING SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully plotted: {successful_plots}/{total_datasets}")
    
    if failed_plots:
        print(f"\n❌ Failed plots ({len(failed_plots)}):")
        for data_name, error in failed_plots:
            print(f"   • {data_name}: {error}")
    
    print(f"\n⏱️  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return successful_plots, failed_plots


def plot_single_dataset(data_name, base_path=None, random_state=2025, 
                       dpi=700, figure_size=(24, 18), show_plot=True):
    """
    Plot a single dataset by name.
    
    Parameters:
    -----------
    data_name : str
        Name of the dataset (e.g., 'mnist', 'pbmc', 'retina')
    base_path : str, optional
        Path to the folder containing embedding files
    random_state : int
        Random state for reproducibility
    dpi : int
        DPI for saved figures
    figure_size : tuple
        Figure size (width, height) in inches
    show_plot : bool
        Whether to display the plot
    """
    
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Find the matching dataset config
    matching_config = None
    for config in DATASETS:
        if config['data_name'] == data_name:
            matching_config = config
            break
    
    if matching_config is None:
        available = ", ".join([d['data_name'] for d in DATASETS])
        raise ValueError(f"Dataset '{data_name}' not found. Available: {available}")
    
    file_path = os.path.join(base_path, matching_config['filename'])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"🔍 Plotting single dataset: {data_name.upper()}")
    print(f"   File: {matching_config['filename']}")
    
    embeddings_dict = np.load(file_path, allow_pickle=True).item()
    
    labels_name = matching_config.get('labels_name', 'Labels')
    exclude_labels = matching_config.get('exclude_labels', None)
    label_names = matching_config.get('label_names', None)
    
    fig, output_path = plot_embeddings_comparison(
        embeddings_dict,
        data_name=data_name,
        random_state=random_state,
        custom_colors=matching_config['custom_colors'],
        exclude_labels=exclude_labels,
        labels_name=labels_name,
        label_names=label_names,
        output_dir=None,
        dpi=dpi,
        figure_size=figure_size,
        point_size=matching_config['point_size'],
        save_plot=True,
        show_plot=show_plot
    )
    
    return fig, output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Plot embedding comparisons for datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Plot a single dataset (e.g., 'mnist', 'pbmc'). If not specified, plots all."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Path to the folder containing embedding files"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=700,
        help="DPI for saved figures (default: 700)"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display plots (only save them)"
    )
    
    args = parser.parse_args()
    
    if args.dataset:
        # Plot single dataset
        try:
            plot_single_dataset(
                args.dataset,
                base_path=args.path,
                dpi=args.dpi,
                show_plot=not args.no_show
            )
        except (ValueError, FileNotFoundError) as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Plot all datasets
        successful, failed = plot_all_datasets(
            base_path=args.path,
            dpi=args.dpi,
            # show_plots=not args.no_show
            show_plots=True
        )
        
        sys.exit(0 if len(failed) == 0 else 1)
