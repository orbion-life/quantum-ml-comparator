"""
Comparison plots and tables for quantum vs classical ML experiments.
Publication-quality figures with matplotlib and seaborn.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


# ============================================================
# Global style configuration
# ============================================================
STYLE_PARAMS = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
    "figure.figsize": (8, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}
plt.rcParams.update(STYLE_PARAMS)
sns.set_palette("Set2")

# Color mapping for model categories
CATEGORY_COLORS = {
    "Quantum": "#8856a7",
    "Classical": "#2ca25f",
    "Hybrid": "#e34a33",
    "Ensemble": "#feb24c",
}

CATEGORY_ORDER = ["Quantum", "Classical", "Hybrid", "Ensemble"]


def _ensure_dir(path):
    """Create parent directories if they don't exist."""
    dirname = os.path.dirname(path) if "." in os.path.basename(path) else path
    if dirname:
        os.makedirs(dirname, exist_ok=True)


def _get_color(model_name, category=None):
    """Get color for a model based on its category."""
    if category and category in CATEGORY_COLORS:
        return CATEGORY_COLORS[category]
    name_lower = model_name.lower()
    if "vqc" in name_lower or "quantum" in name_lower or "kernel" in name_lower:
        return CATEGORY_COLORS["Quantum"]
    elif "hybrid" in name_lower:
        return CATEGORY_COLORS["Hybrid"]
    elif "ensemble" in name_lower:
        return CATEGORY_COLORS["Ensemble"]
    else:
        return CATEGORY_COLORS["Classical"]


def _save_or_show(fig, save_path):
    """Save figure to path if provided, otherwise show it."""
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Tables
# ============================================================

def generate_comparison_table(all_results):
    """
    Create a pandas DataFrame with all models ranked by F1, grouped by category.

    Parameters
    ----------
    all_results : dict
        Mapping of model_name -> {
            'metrics': dict from compute_metrics,
            'category': str ('Quantum', 'Classical', 'Hybrid', 'Ensemble'),
            'n_params': int (optional),
            'training_time_s': float (optional),
        }

    Returns
    -------
    pd.DataFrame
        Sorted by F1 (macro) descending, with category grouping.
    """
    rows = []
    for model_name, data in all_results.items():
        if data is None or not isinstance(data, dict):
            continue
        m = data.get("metrics", data)
        if m is None:
            continue
        row = {
            "Model": model_name,
            "Category": data.get("category", "Unknown"),
            "F1 (macro)": m.get("f1_macro", 0.0),
            "F1 (weighted)": m.get("f1_weighted", 0.0),
            "Accuracy": m.get("accuracy", 0.0),
            "Precision": m.get("precision", 0.0),
            "Recall": m.get("recall", 0.0),
            "AUC-ROC": m.get("auc_roc", None),
            "MCC": m.get("mcc", 0.0),
            "Parameters": data.get("n_params", None),
            "Train Time (s)": data.get("training_time_s", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by category order, then by F1 descending within each category
    cat_rank = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    df["_cat_rank"] = df["Category"].map(lambda x: cat_rank.get(x, 99))
    df = df.sort_values(["_cat_rank", "F1 (macro)"], ascending=[True, False])
    df = df.drop(columns=["_cat_rank"])
    df = df.reset_index(drop=True)

    # Add rank column
    df.insert(
        0, "Rank", df["F1 (macro)"].rank(ascending=False, method="min").astype(int)
    )

    return df


# ============================================================
# Plot functions
# ============================================================

def plot_f1_comparison(all_results, save_path=None):
    """
    Bar chart of F1 scores, color-coded by category.

    Parameters
    ----------
    all_results : dict
        Mapping of model_name -> result dict (see generate_comparison_table).
    save_path : str or None
        File path to save the figure. If None, the plot is displayed.
    """
    models = []
    f1_scores = []
    colors = []
    categories = []

    for model_name, data in all_results.items():
        m = data.get("metrics", data)
        models.append(model_name)
        f1_scores.append(m.get("f1_macro", 0.0))
        category = data.get("category", "Classical")
        categories.append(category)
        colors.append(_get_color(model_name, category))

    # Sort by F1
    order = np.argsort(f1_scores)[::-1]
    models = [models[i] for i in order]
    f1_scores = [f1_scores[i] for i in order]
    colors = [colors[i] for i in order]
    categories = [categories[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 0.8), 6))
    bars = ax.bar(
        range(len(models)), f1_scores, color=colors, edgecolor="white", linewidth=0.5
    )

    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Model Comparison: F1 Scores by Category")
    ax.set_ylim(0, min(1.05, max(f1_scores) * 1.15) if f1_scores else 1.05)

    # Category legend
    legend_elements = [
        Line2D([0], [0], color=c, lw=6, label=cat)
        for cat, c in CATEGORY_COLORS.items()
        if cat in set(categories)
    ]
    if legend_elements:
        ax.legend(
            handles=legend_elements, loc="upper right", title="Category", framealpha=0.9
        )

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curves(all_results, save_path=None):
    """
    Overlaid ROC curves for models that provide probability outputs.

    Parameters
    ----------
    all_results : dict
        Mapping of model_name -> result dict. Must include 'roc_curve' key
        with (fpr, tpr) arrays, or 'y_prob' and 'y_true' for on-the-fly
        computation.
    save_path : str or None
        File path to save the figure. If None, the plot is displayed.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 7))

    for model_name, data in all_results.items():
        category = data.get("category", "Classical")
        color = _get_color(model_name, category)
        m = data.get("metrics", data)
        auc_val = m.get("auc_roc")

        if "roc_curve" in data:
            fpr, tpr = data["roc_curve"]
        elif "y_prob" in data and "y_true" in data:
            y_prob = np.asarray(data["y_prob"])
            y_true = np.asarray(data["y_true"])
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = auc(fpr, tpr)
        else:
            continue

        label = f"{model_name} (AUC={auc_val:.3f})" if auc_val else model_name
        ax.plot(fpr, tpr, color=color, linewidth=1.8, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_confusion_matrices(all_results, save_path=None, models=None):
    """
    Side-by-side confusion matrices for selected models.

    Parameters
    ----------
    all_results : dict
        Mapping of model_name -> result dict with 'metrics' containing
        'confusion_matrix'.
    save_path : str or None
        File path to save. If None, the plot is displayed.
    models : list of str or None
        Models to include. If None, picks top 4 by F1.
    """
    if models is None:
        ranked = sorted(
            all_results.items(),
            key=lambda x: x[1].get("metrics", x[1]).get("f1_macro", 0),
            reverse=True,
        )
        models = [name for name, _ in ranked[:4]]

    n = len(models)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, model_name in enumerate(models):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        data = all_results.get(model_name, {})
        m = data.get("metrics", data)
        cm = np.array(m.get("confusion_matrix", [[0]]))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            square=True,
            linewidths=0.5,
        )
        f1 = m.get("f1_macro", 0.0)
        ax.set_title(f"{model_name}\nF1={f1:.3f}", fontsize=10)
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_learning_curves(learning_curve_data, save_path=None):
    """
    F1 vs dataset size for each model.

    Parameters
    ----------
    learning_curve_data : dict
        Mapping of model_name -> {
            'sizes': list, 'f1_mean': list, 'f1_std': list
        }
    save_path : str or None
        File path to save. If None, the plot is displayed.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for model_name, data in learning_curve_data.items():
        sizes = data["sizes"]
        f1_mean = np.array(data["f1_mean"])
        f1_std = np.array(data["f1_std"])
        color = _get_color(model_name)

        ax.plot(
            sizes,
            f1_mean,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            label=model_name,
        )
        ax.fill_between(
            sizes, f1_mean - f1_std, f1_mean + f1_std, color=color, alpha=0.15
        )

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Learning Curves: F1 vs Training Data Size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    plt.tight_layout()
    _save_or_show(fig, save_path)
