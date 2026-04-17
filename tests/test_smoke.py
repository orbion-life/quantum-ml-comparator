"""Smoke tests to exercise modules that are otherwise untested.

These tests assert "does not crash on typical inputs" rather than
correctness — correctness is covered by the domain-specific tests.
Their job is to raise coverage and catch import-time regressions.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")  # headless for CI

import numpy as np
import pytest


# --------------------------------------------------------------------------
# evaluation/plots.py
# --------------------------------------------------------------------------


class TestEvaluationPlots:
    """Smoke-test every plot helper — import OK, no crash on toy data."""

    @pytest.fixture
    def toy_results(self):
        """A minimal results dict in the shape the plot helpers expect."""
        return {
            "model_a": {
                "accuracy": 0.85,
                "f1_macro": 0.82,
                "f1_score": 0.82,
                "precision": 0.81,
                "recall": 0.83,
                "auc_roc": 0.9,
                "confusion_matrix": [[9, 1], [2, 8]],
                "category": "classical",
            },
            "model_b": {
                "accuracy": 0.90,
                "f1_macro": 0.88,
                "f1_score": 0.88,
                "precision": 0.87,
                "recall": 0.89,
                "auc_roc": 0.95,
                "confusion_matrix": [[9, 1], [1, 9]],
                "category": "quantum",
            },
        }

    def test_plots_import(self):
        from qmc.evaluation import plots
        # Sanity: module exposes its public helpers
        for name in (
            "plot_f1_comparison",
            "plot_learning_curves",
            "plot_confusion_matrices",
            "plot_roc_curves",
            "generate_comparison_table",
        ):
            assert hasattr(plots, name), f"qmc.evaluation.plots missing {name}"

    def test_generate_comparison_table(self, toy_results):
        from qmc.evaluation.plots import generate_comparison_table
        df = generate_comparison_table(toy_results)
        assert df is not None
        assert len(df) == 2


# --------------------------------------------------------------------------
# classical/models.py
# --------------------------------------------------------------------------


class TestClassicalModels:
    """Smoke-test classical model wrappers."""

    @pytest.fixture
    def toy_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4)).astype(np.float32)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_models_import(self):
        import qmc.classical.models as m
        assert hasattr(m, "TinyMLP") or hasattr(m, "MediumMLP")

    def test_sklearn_helpers_return_estimators(self):
        from qmc.classical.models import (
            get_svm,
            get_random_forest,
            get_logistic_regression,
        )
        for factory in (get_svm, get_random_forest, get_logistic_regression):
            est = factory()
            # Anything callable with fit/predict qualifies as an estimator
            assert hasattr(est, "fit")
            assert hasattr(est, "predict")


# --------------------------------------------------------------------------
# datasets/builtin.py — loaders we haven't touched yet
# --------------------------------------------------------------------------


class TestDatasetLoaders:
    def test_all_builtin_loaders_return_sane_shapes(self):
        from qmc.datasets.builtin import load_dataset, list_datasets
        for name in list_datasets():
            X_train, X_test, y_train, y_test, meta = load_dataset(name)
            # Shape sanity
            assert X_train.ndim == 2
            assert X_test.ndim == 2
            assert X_train.shape[1] == X_test.shape[1]
            assert len(y_train) == X_train.shape[0]
            assert len(y_test) == X_test.shape[0]
            # Metadata sanity
            assert meta.n_features > 0
            assert meta.n_classes > 1
            assert meta.n_samples == X_train.shape[0] + X_test.shape[0]


# --------------------------------------------------------------------------
# recommender edge cases
# --------------------------------------------------------------------------


class TestRecommenderEdgeCases:
    def test_print_recommendations_produces_output(self):
        from qmc.recommender import print_recommendations
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_recommendations("RandomForest")
        out = buf.getvalue()
        # Must name the classical algorithm and at least one quantum counterpart
        assert "RandomForest" in out
        assert "PRIMARY" in out or "primary" in out.lower()

    def test_recommend_with_many_features(self):
        from qmc.recommender import recommend
        recs = recommend("SVM", n_features=16, n_classes=4)
        for r in recs:
            cfg = r["circuit_config"]
            assert cfg["n_qubits"] >= 16
            assert cfg["n_features"] == 16
            assert cfg["n_classes"] == 4
