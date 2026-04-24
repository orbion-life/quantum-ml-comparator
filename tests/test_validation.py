"""Tests for the pydantic-backed input validation of the public ``qmc`` API.

These tests prove that bad input is *rejected at the trust boundary* with
a clear error, rather than silently propagating into the math kernels.
This is the evidence the ISO 27001 sanitized-inputs-code-scanning audit
needs.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from qmc import (
    Benchmark,
    FeatureChannelBenchmark,
    QuantumKernelClassifier,
    VQCClassifier,
    print_recommendations,
    recommend,
)


# ---------------------------------------------------------------------------
# recommend() / print_recommendations()
# ---------------------------------------------------------------------------


class TestRecommendInputValidation:
    def test_accepts_known_algorithm(self):
        recs = recommend("SVM")
        assert len(recs) > 0

    def test_strips_and_normalises(self):
        # str_strip_whitespace=True on RecommendInput
        assert recommend("  SVM  ") == recommend("SVM")

    def test_rejects_empty_algorithm(self):
        with pytest.raises(ValidationError):
            recommend("")

    def test_rejects_non_string_algorithm(self):
        with pytest.raises(ValidationError):
            recommend(123)  # type: ignore[arg-type]

    def test_rejects_overlong_algorithm(self):
        with pytest.raises(ValidationError):
            recommend("x" * 1000)

    def test_rejects_n_features_below_one(self):
        with pytest.raises(ValidationError):
            recommend("SVM", n_features=0)

    def test_rejects_n_features_above_max(self):
        with pytest.raises(ValidationError):
            recommend("SVM", n_features=10**6)

    def test_rejects_n_classes_below_two(self):
        with pytest.raises(ValidationError):
            recommend("SVM", n_classes=1)

    def test_print_recommendations_validates_too(self, capsys):
        with pytest.raises(ValidationError):
            print_recommendations("", n_features=8, n_classes=2)
        # Sanity: a valid call still works
        print_recommendations("SVM")
        out = capsys.readouterr().out
        assert "SVM" in out


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class TestBenchmarkValidation:
    def test_accepts_builtin_dataset(self):
        Benchmark(dataset="iris", quantum_methods=["VQC"], classical_methods=["MLP"])

    def test_accepts_array_tuple(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(20, 4)), rng.integers(0, 2, size=20)
        Benchmark(dataset=(X, y), quantum_methods=["VQC"], classical_methods=["MLP"])

    def test_rejects_unknown_dataset_type(self):
        with pytest.raises(TypeError):
            Benchmark(dataset=123)  # type: ignore[arg-type]

    def test_rejects_dataset_path_that_looks_like_path_but_missing(self):
        with pytest.raises(ValueError, match="no such file"):
            Benchmark(dataset="/no/such/file.csv")

    def test_rejects_dataset_tuple_of_wrong_length(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 4))
        y = rng.integers(0, 2, size=10)  # mismatched
        with pytest.raises(ValueError, match="rows but y has"):
            Benchmark(dataset=(X, y))

    def test_rejects_dataset_with_nan(self):
        X = np.full((10, 3), np.nan)
        y = np.zeros(10)
        with pytest.raises(ValueError, match="non-finite"):
            Benchmark(dataset=(X, y))

    def test_rejects_negative_n_qubits(self):
        with pytest.raises(ValidationError):
            Benchmark(dataset="iris", n_qubits=-1)

    def test_rejects_n_qubits_above_max(self):
        with pytest.raises(ValidationError):
            Benchmark(dataset="iris", n_qubits=10_000)

    def test_rejects_test_size_outside_open_interval(self):
        with pytest.raises(ValidationError):
            Benchmark(dataset="iris", test_size=0.0)
        with pytest.raises(ValidationError):
            Benchmark(dataset="iris", test_size=1.0)

    def test_rejects_negative_random_state(self):
        with pytest.raises(ValidationError):
            Benchmark(dataset="iris", random_state=-1)

    def test_rejects_empty_classical_methods(self):
        with pytest.raises(ValueError, match="non-empty"):
            Benchmark(dataset="iris", classical_methods=[])

    def test_rejects_duplicate_quantum_methods(self):
        with pytest.raises(ValueError, match="duplicate"):
            Benchmark(dataset="iris", quantum_methods=["VQC", "VQC"])

    def test_rejects_non_string_method(self):
        with pytest.raises(ValueError, match="must be a string"):
            Benchmark(dataset="iris", quantum_methods=["VQC", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# FeatureChannelBenchmark
# ---------------------------------------------------------------------------


class TestFeatureChannelBenchmarkValidation:
    def _make_channels(self, n_tr=20, n_te=10, dim=4):
        rng = np.random.default_rng(0)
        X_tr = rng.normal(size=(n_tr, dim))
        X_te = rng.normal(size=(n_te, dim))
        y_tr = rng.integers(0, 2, size=n_tr)
        y_te = rng.integers(0, 2, size=n_te)
        return {"a": (X_tr, X_te)}, y_tr, y_te

    def test_happy_path(self):
        chans, y_tr, y_te = self._make_channels()
        FeatureChannelBenchmark(
            channels=chans, y_train=y_tr, y_test=y_te,
            estimator_factory=lambda: object(),
        )

    def test_rejects_non_callable_factory(self):
        chans, y_tr, y_te = self._make_channels()
        with pytest.raises(TypeError, match="callable"):
            FeatureChannelBenchmark(
                channels=chans, y_train=y_tr, y_test=y_te,
                estimator_factory="not callable",  # type: ignore[arg-type]
            )

    def test_rejects_negative_seed(self):
        chans, y_tr, y_te = self._make_channels()
        with pytest.raises(ValidationError):
            FeatureChannelBenchmark(
                channels=chans, y_train=y_tr, y_test=y_te,
                estimator_factory=lambda: object(), seed=-1,
            )

    def test_rejects_empty_training_sizes(self):
        chans, y_tr, y_te = self._make_channels()
        with pytest.raises(ValueError, match="non-empty"):
            FeatureChannelBenchmark(
                channels=chans, y_train=y_tr, y_test=y_te,
                estimator_factory=lambda: object(), training_sizes=[],
            )

    def test_rejects_training_size_exceeding_train_rows(self):
        chans, y_tr, y_te = self._make_channels(n_tr=20)
        with pytest.raises(ValueError, match="exceeds available train rows"):
            FeatureChannelBenchmark(
                channels=chans, y_train=y_tr, y_test=y_te,
                estimator_factory=lambda: object(), training_sizes=[100],
            )

    def test_rejects_non_integer_training_size(self):
        chans, y_tr, y_te = self._make_channels()
        with pytest.raises(ValueError, match="must be an integer"):
            FeatureChannelBenchmark(
                channels=chans, y_train=y_tr, y_test=y_te,
                estimator_factory=lambda: object(), training_sizes=[1.5],  # type: ignore[list-item]
            )


# ---------------------------------------------------------------------------
# Sklearn-API classifier hyperparameter validation
# ---------------------------------------------------------------------------


class TestVQCClassifierValidation:
    def test_init_does_not_validate(self):
        # Per sklearn contract __init__ is store-only, so bad params are
        # only caught at .fit() time. This test pins that contract.
        VQCClassifier(n_qubits=-1, n_layers=-1, epochs=0, lr=-1.0)

    def test_fit_rejects_negative_qubits(self):
        clf = VQCClassifier(n_qubits=-1)
        X = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
        with pytest.raises(ValidationError):
            clf.fit(X, y)

    def test_fit_rejects_zero_epochs(self):
        clf = VQCClassifier(epochs=0)
        X = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
        with pytest.raises(ValidationError):
            clf.fit(X, y)

    def test_fit_rejects_non_finite_lr(self):
        clf = VQCClassifier(lr=float("nan"))
        X = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
        with pytest.raises(ValidationError):
            clf.fit(X, y)


class TestQuantumKernelClassifierValidation:
    def test_fit_rejects_negative_qubits(self):
        clf = QuantumKernelClassifier(n_qubits=-1)
        X = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
        with pytest.raises(ValidationError):
            clf.fit(X, y)

    def test_fit_rejects_negative_seed(self):
        clf = QuantumKernelClassifier(seed=-1)
        X = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
        with pytest.raises(ValidationError):
            clf.fit(X, y)
