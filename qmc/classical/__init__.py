"""Classical ML models for comparison with quantum approaches."""

from qmc.classical.models import (
    TinyMLP,
    MediumMLP,
    get_svm,
    get_random_forest,
    get_logistic_regression,
    train_pytorch_model,
    evaluate_model,
    count_params,
)

__all__ = [
    "TinyMLP",
    "MediumMLP",
    "get_svm",
    "get_random_forest",
    "get_logistic_regression",
    "train_pytorch_model",
    "evaluate_model",
    "count_params",
]
