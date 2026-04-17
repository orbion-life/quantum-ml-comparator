"""QML algorithm recommender.

Given a classical ML algorithm name, return ranked quantum ML counterparts
with rationale, difficulty level, and auto-configured circuit parameters.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize(name: str) -> str:
    """Collapse a user-supplied algorithm name to a canonical lowercase key.

    Handles aliases like "random_forest", "RandomForest", "RF",
    "random forest", "random-forest" -> "random_forest".
    """
    s = name.strip().lower()
    # Replace dashes and spaces with underscores
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def _circuit_config(
    n_features: int,
    n_classes: int,
    *,
    n_layers: Optional[int] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Build a circuit_config dict sized to the problem."""
    n_qubits = max(2, n_features)
    if n_layers is None:
        # Heuristic: ceil(log2(n_features)) + 1, minimum 2
        n_layers = max(2, math.ceil(math.log2(max(n_features, 2))) + 1)
    config = {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "n_features": n_features,
        "n_classes": n_classes,
    }
    if extra:
        config.update(extra)
    return config


# ---------------------------------------------------------------------------
# Mapping table
# ---------------------------------------------------------------------------

# Each entry: list of dicts ordered by priority (primary first).
# Fields filled at lookup time: circuit_config, classical_analog.

_MAPPINGS: Dict[str, List[dict]] = {
    # --- SVM -----------------------------------------------------------
    "svm": [
        {
            "name": "Quantum Kernel SVM",
            "description": (
                "Replace the classical kernel with a quantum feature-map "
                "kernel evaluated on a quantum device, then train a "
                "classical SVM on the resulting kernel matrix."
            ),
            "rationale": (
                "Quantum kernels can express feature maps that are hard "
                "to compute classically, potentially capturing non-linear "
                "decision boundaries more efficiently."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"kernel": "ZZFeatureMap"},
        },
        {
            "name": "VQC",
            "description": (
                "Variational Quantum Classifier: a parameterised quantum "
                "circuit trained end-to-end with classical optimisation."
            ),
            "rationale": (
                "VQC acts as a trainable quantum neural network and can "
                "serve as a drop-in replacement for SVM on small datasets."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"ansatz": "StronglyEntangling"},
        },
    ],

    # --- MLP / Neural Network ------------------------------------------
    "mlp": [
        {
            "name": "VQC",
            "description": (
                "Variational Quantum Classifier: a parameterised quantum "
                "circuit trained end-to-end, analogous to a shallow "
                "neural network."
            ),
            "rationale": (
                "A VQC with multiple entangling layers mirrors the "
                "layer-wise structure of an MLP and can learn similar "
                "non-linear mappings."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"ansatz": "StronglyEntangling"},
        },
        {
            "name": "Data Re-uploading VQC",
            "description": (
                "A VQC variant where input data is re-encoded into the "
                "circuit at every layer, increasing expressibility."
            ),
            "rationale": (
                "Re-uploading data at each layer is analogous to skip "
                "connections in deep networks and can improve "
                "representation power on structured data."
            ),
            "difficulty": "medium",
            "_layers": None,
            "_extra": {"ansatz": "DataReUploading", "re_upload": True},
        },
    ],

    # --- Random Forest -------------------------------------------------
    "random_forest": [
        {
            "name": "Quantum Kernel Ensemble",
            "description": (
                "Ensemble of quantum-kernel SVMs, each trained on a "
                "bootstrap sample of the data, mimicking the bagging "
                "strategy of Random Forest."
            ),
            "rationale": (
                "Combining multiple quantum kernel models reduces "
                "variance, similar to how Random Forest aggregates "
                "decision trees."
            ),
            "difficulty": "medium",
            "_layers": None,
            "_extra": {"kernel": "ZZFeatureMap", "n_estimators": 10},
        },
        {
            "name": "VQC",
            "description": (
                "Variational Quantum Classifier used as a single strong "
                "learner to replace the tree ensemble."
            ),
            "rationale": (
                "While Random Forest relies on many weak learners, a "
                "sufficiently expressive VQC can achieve competitive "
                "accuracy as a single model."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"ansatz": "StronglyEntangling"},
        },
    ],

    # --- Logistic Regression -------------------------------------------
    "logistic_regression": [
        {
            "name": "Quantum Kernel + Linear SVM",
            "description": (
                "Project features into a quantum Hilbert space via a "
                "feature map, then fit a linear SVM on the kernel matrix."
            ),
            "rationale": (
                "Logistic regression is a linear classifier; a quantum "
                "kernel lifts features into a high-dimensional space "
                "where a linear boundary may suffice."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"kernel": "ZZFeatureMap", "svm_type": "linear"},
        },
        {
            "name": "Single-layer VQC",
            "description": (
                "A minimal VQC with a single variational layer, "
                "functioning as a quantum logistic regression."
            ),
            "rationale": (
                "One variational layer keeps model complexity comparable "
                "to logistic regression while exploiting quantum "
                "feature encoding."
            ),
            "difficulty": "easy",
            "_layers": 1,
            "_extra": {"ansatz": "BasicEntangler"},
        },
    ],

    # --- k-NN ----------------------------------------------------------
    "knn": [
        {
            "name": "Quantum Kernel k-NN",
            "description": (
                "Use a quantum kernel to define distances between "
                "data points, then apply the k-nearest-neighbours rule "
                "in quantum-enhanced feature space."
            ),
            "rationale": (
                "Quantum kernels can capture complex similarity "
                "structures that Euclidean distance may miss, "
                "potentially improving neighbour quality."
            ),
            "difficulty": "medium",
            "_layers": None,
            "_extra": {"kernel": "ZZFeatureMap", "n_neighbors": 5},
        },
    ],

    # --- XGBoost / Gradient Boosting -----------------------------------
    "xgboost": [
        {
            "name": "Quantum Kernel SVM",
            "description": (
                "Quantum kernel SVM as a strong single learner, "
                "replacing the boosted ensemble."
            ),
            "rationale": (
                "Gradient boosting builds accuracy iteratively; a "
                "quantum kernel SVM can instead leverage a "
                "high-dimensional feature space in a single shot."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"kernel": "ZZFeatureMap"},
        },
        {
            "name": "Quantum Boosted Ensemble",
            "description": (
                "Boosting framework where each weak learner is a small "
                "VQC, combined via classical boosting logic."
            ),
            "rationale": (
                "Preserves the sequential error-correction of boosting "
                "while letting each stage exploit quantum expressibility."
            ),
            "difficulty": "hard",
            "_layers": 2,
            "_extra": {"ansatz": "BasicEntangler", "n_estimators": 10, "boosting": True},
        },
    ],

    # --- Naive Bayes ---------------------------------------------------
    "naive_bayes": [
        {
            "name": "VQC with probabilistic readout",
            "description": (
                "A VQC whose measurement probabilities are interpreted "
                "as class posterior estimates, analogous to Naive Bayes "
                "posterior output."
            ),
            "rationale": (
                "Quantum measurement is inherently probabilistic, "
                "making it a natural fit for probabilistic classifiers."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"ansatz": "StronglyEntangling", "readout": "probs"},
        },
    ],

    # --- PCA -----------------------------------------------------------
    "pca": [
        {
            "name": "Quantum feature map",
            "description": (
                "Encode data with a quantum feature map and measure "
                "in a reduced qubit subspace to achieve dimensionality "
                "reduction."
            ),
            "rationale": (
                "Quantum feature maps project data into an exponentially "
                "large Hilbert space; partial measurement acts as a "
                "non-linear dimensionality reducer."
            ),
            "difficulty": "easy",
            "_layers": None,
            "_extra": {"kernel": "AngleEmbedding", "task": "dim_reduction"},
        },
        {
            "name": "Quantum Autoencoder",
            "description": (
                "A variational circuit trained to compress quantum states "
                "into fewer qubits, then reconstruct the input — the "
                "quantum analogue of an autoencoder."
            ),
            "rationale": (
                "Quantum autoencoders learn a compact latent space on a "
                "quantum device, offering potential advantages when data "
                "has quantum-native structure."
            ),
            "difficulty": "medium",
            "_layers": None,
            "_extra": {"ansatz": "StronglyEntangling", "task": "dim_reduction",
                       "latent_qubits": 2},
        },
    ],
}

# General-purpose quantum methods (fallback for unmapped algorithms)
_GENERAL_PURPOSE: List[dict] = [
    {
        "name": "VQC",
        "description": (
            "Variational Quantum Classifier: a general-purpose "
            "parameterised quantum circuit for classification."
        ),
        "rationale": (
            "VQC is the most broadly applicable QML model and serves "
            "as a reasonable starting point for any classification task."
        ),
        "difficulty": "easy",
        "_layers": None,
        "_extra": {"ansatz": "StronglyEntangling"},
    },
    {
        "name": "Quantum Kernel",
        "description": (
            "Compute a kernel matrix using a quantum feature map and "
            "feed it to a classical kernel method (SVM, k-NN, etc.)."
        ),
        "rationale": (
            "Quantum kernels are model-agnostic and can enhance any "
            "classical method that accepts a precomputed kernel."
        ),
        "difficulty": "easy",
        "_layers": None,
        "_extra": {"kernel": "ZZFeatureMap"},
    },
]

# Alias table: maps normalised alias -> canonical key in _MAPPINGS
_ALIASES: Dict[str, str] = {
    # SVM
    "svm": "svm",
    "svc": "svm",
    "support_vector_machine": "svm",
    "support_vector_classifier": "svm",
    # MLP / Neural Network
    "mlp": "mlp",
    "neural_network": "mlp",
    "nn": "mlp",
    "neural_net": "mlp",
    "multi_layer_perceptron": "mlp",
    "perceptron": "mlp",
    "deep_learning": "mlp",
    "dl": "mlp",
    "ann": "mlp",
    # Random Forest
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "rf": "random_forest",
    # Logistic Regression
    "logistic_regression": "logistic_regression",
    "logisticregression": "logistic_regression",
    "logreg": "logistic_regression",
    "lr": "logistic_regression",
    # k-NN
    "knn": "knn",
    "k_nn": "knn",
    "k_nearest_neighbors": "knn",
    "k_nearest_neighbours": "knn",
    "nearest_neighbors": "knn",
    "nearest_neighbours": "knn",
    # XGBoost / Gradient Boosting
    "xgboost": "xgboost",
    "xgb": "xgboost",
    "gradient_boosting": "xgboost",
    "gradientboosting": "xgboost",
    "gbm": "xgboost",
    "gradient_boosted_trees": "xgboost",
    "lightgbm": "xgboost",
    "lgbm": "xgboost",
    "catboost": "xgboost",
    "boosting": "xgboost",
    # Naive Bayes
    "naive_bayes": "naive_bayes",
    "naivebayes": "naive_bayes",
    "nb": "naive_bayes",
    "gaussian_nb": "naive_bayes",
    "gaussiannb": "naive_bayes",
    # PCA
    "pca": "pca",
    "principal_component_analysis": "pca",
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend(
    classical_algorithm: str,
    n_features: int = 8,
    n_classes: int = 2,
) -> list[dict]:
    """Return ranked quantum ML counterparts for *classical_algorithm*.

    Parameters
    ----------
    classical_algorithm:
        Name of a classical ML algorithm (case-insensitive, alias-tolerant).
    n_features:
        Number of input features; used to size circuit configs.
    n_classes:
        Number of target classes.

    Returns
    -------
    list[dict]
        Each dict contains: name, description, rationale, difficulty,
        circuit_config, classical_analog.
    """
    key = _normalize(classical_algorithm)
    canonical = _ALIASES.get(key)

    if canonical is not None:
        templates = _MAPPINGS[canonical]
    else:
        # Fallback: general-purpose quantum methods
        templates = _GENERAL_PURPOSE

    results: list[dict] = []
    for tmpl in templates:
        cfg = _circuit_config(
            n_features,
            n_classes,
            n_layers=tmpl.get("_layers"),
            extra=tmpl.get("_extra"),
        )
        results.append(
            {
                "name": tmpl["name"],
                "description": tmpl["description"],
                "rationale": tmpl["rationale"],
                "difficulty": tmpl["difficulty"],
                "circuit_config": cfg,
                "classical_analog": classical_algorithm,
            }
        )
    return results


def print_recommendations(
    classical_algorithm: str,
    n_features: int = 8,
    n_classes: int = 2,
) -> None:
    """Pretty-print quantum ML recommendations to stdout."""
    recs = recommend(classical_algorithm, n_features, n_classes)
    header = (
        f"Quantum ML recommendations for '{classical_algorithm}' "
        f"(features={n_features}, classes={n_classes})"
    )
    print(header)
    print("=" * len(header))
    for i, rec in enumerate(recs, 1):
        tag = "PRIMARY" if i == 1 else "SECONDARY"
        print(f"\n  [{tag}] {rec['name']}  (difficulty: {rec['difficulty']})")
        print(f"    {rec['description']}")
        print(f"    Rationale : {rec['rationale']}")
        cfg = rec["circuit_config"]
        print(f"    Circuit   : {cfg['n_qubits']} qubits, {cfg['n_layers']} layers")
        extras = {k: v for k, v in cfg.items()
                  if k not in ("n_qubits", "n_layers", "n_features", "n_classes")}
        if extras:
            print(f"    Extras    : {extras}")
    print()


def get_all_mappings() -> dict:
    """Return the full mapping table (canonical keys -> template lists).

    Also includes a ``"_general_purpose"`` key for the fallback list and
    an ``"_aliases"`` key for the alias table.
    """
    return {
        **{k: v for k, v in _MAPPINGS.items()},
        "_general_purpose": list(_GENERAL_PURPOSE),
        "_aliases": dict(_ALIASES),
    }
