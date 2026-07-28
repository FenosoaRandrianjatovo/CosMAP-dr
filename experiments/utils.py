from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_all_datasets import plot_embeddings_comparison





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
RETINA_COLORS = {
    "BC1A": "red", "BC1B": "black", "BC2": "blue", "BC3A": "#D191EB",
    "BC3B": "magenta", "BC4": "cyan", "BC5A": "green", "BC5B": "purple",
    "BC5C": "pink", "BC5D": "brown", "BC6": "gray", "BC7": "navy",
    "BC8_9": "lightgreen", "MG": "maroon", "RBC": "orange"
}

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


# Only options that differ between datasets
PLOT_CONFIG = {
    "retina": {
        "custom_colors": RETINA_COLORS,
        "labels_name": "Cell type",
        "exclude_labels":None
    },
    "cortex": {
        "custom_colors": None,
        "labels_name": "Cell type",
        "exclude_labels":None
    },
    "pbmc": {
        "custom_colors": PBMC_COLORS,
        "labels_name": "Cell type",
        "exclude_labels":None
    },
    "paul15": {
        "custom_colors": None,
        "labels_name": "Cell type",
        "exclude_labels":None
    },
    "fmnist": {
        "custom_colors": None,
        "labels_name": "Fashion class",
        "exclude_labels":None
    },
    "coil_20": {
        "custom_colors": None,
        "labels_name": "Object class",
        "exclude_labels":None
    },
    "20NG": {
        "custom_colors": None,
        "labels_name": "Newsgroup",
        "exclude_labels":None
    },
    "mnist": {
        "custom_colors": None,
        "labels_name": "Digit class",
        "exclude_labels":None
    },
    "USPS": {
        "custom_colors": None,
        "labels_name": "Digit class",
        "exclude_labels":None
    },
    "heart_cell_atlas": {
        "custom_colors": None,
        "labels_name": "Cell type",
        "exclude_labels":None
    },
    "kinship_rrq": {
        "custom_colors": None,
        "labels_name": "Region",
        "exclude_labels":None
    },
    "kinship_cartagene": {
        "custom_colors": KINSHIP_COLORS,
        "labels_name": "Parents' marriage region",
        "exclude_labels":["Unknown"]
    },
}