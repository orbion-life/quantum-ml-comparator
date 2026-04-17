"""Evaluation metrics and visualization for model comparison."""

from qmc.evaluation.metrics import compute_metrics, compare_models
from qmc.evaluation.plots import (
    plot_f1_comparison,
    plot_learning_curves,
    plot_confusion_matrices,
    plot_roc_curves,
    generate_comparison_table,
)

__all__ = [
    "compute_metrics",
    "compare_models",
    "plot_f1_comparison",
    "plot_learning_curves",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "generate_comparison_table",
]
