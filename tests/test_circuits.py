"""Tests for quantum circuit modules."""
import pytest
import numpy as np


class TestCircuitTemplates:
    def test_create_device(self):
        from qmc.circuits.templates import create_device
        dev = create_device(4)
        assert dev is not None

    def test_weight_shapes(self):
        from qmc.circuits.templates import get_weight_shapes
        shapes = get_weight_shapes(4, 2)
        assert "weights" in shapes
        assert shapes["weights"] == (2, 4, 3)


class TestQNP:
    def test_param_shape(self):
        from qmc.circuits.qnp import get_qnp_param_shape, count_qnp_params
        shape = get_qnp_param_shape(8, 4)
        assert len(shape) == 3
        n = count_qnp_params(8, 4)
        assert n > 0
        assert n == np.prod(shape)

    def test_initialize_params(self):
        from qmc.circuits.qnp import initialize_qnp_params
        params = initialize_qnp_params(8, 4, strategy="A")
        assert params is not None


class TestMetrics:
    def test_compute_metrics(self):
        from qmc.evaluation.metrics import compute_metrics
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        m = compute_metrics(y_true, y_pred)
        assert "accuracy" in m
        assert "f1_macro" in m
        assert 0 <= m["accuracy"] <= 1

    def test_compare_models(self):
        from qmc.evaluation.metrics import compute_metrics, compare_models
        # Use actual compute_metrics to get full metric dict
        y_true = np.array([0, 0, 1, 1, 1])
        pred_a = np.array([0, 1, 1, 1, 0])  # lower F1
        pred_b = np.array([0, 0, 1, 1, 1])  # perfect
        results = {
            "model_a": compute_metrics(y_true, pred_a),
            "model_b": compute_metrics(y_true, pred_b),
        }
        comparison = compare_models(results)
        assert "ranking" in comparison
        assert comparison["ranking"][0] == "model_b"
