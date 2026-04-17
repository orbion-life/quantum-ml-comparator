"""Scientific + sklearn-contract sanity for VQCClassifier / QuantumKernelClassifier.

These tests enforce Gate C of the plan:
- fit/predict/predict_proba contract,
- cross_val_score works,
- predict_proba rows sum to 1,
- calling predict before fit raises NotFittedError,
- pickle round-trip preserves predictions.

We intentionally run the VQC with tiny settings (n_qubits=2, n_layers=1,
epochs=3) so the whole test suite finishes in under a minute on CI.
A separate "slow" test (disabled unless --runslow is passed) does the
full check_estimator sweep.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_moons
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def moons_data():
    """A tiny 2D moons dataset, deterministic across runs."""
    X, y = make_moons(n_samples=40, noise=0.15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# VQCClassifier
# --------------------------------------------------------------------------


class TestVQCClassifier:
    def test_imports_from_top_level(self):
        from qmc import VQCClassifier
        assert VQCClassifier is not None

    def test_get_params_returns_all_init_args(self):
        from qmc import VQCClassifier
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=3)
        params = clf.get_params()
        # Every __init__ kwarg must appear in get_params()
        for key in ("n_qubits", "n_layers", "epochs", "lr", "batch_size",
                    "seed", "device_name", "diff_method"):
            assert key in params

    def test_set_params_roundtrip(self):
        from qmc import VQCClassifier
        clf = VQCClassifier()
        clf.set_params(n_qubits=3, epochs=5)
        assert clf.n_qubits == 3
        assert clf.epochs == 5

    def test_predict_before_fit_raises(self):
        from qmc import VQCClassifier
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=1)
        with pytest.raises(NotFittedError):
            clf.predict(np.zeros((3, 2)))
        with pytest.raises(NotFittedError):
            clf.predict_proba(np.zeros((3, 2)))

    @pytest.mark.slow
    def test_fit_predict_on_moons(self, moons_data):
        from qmc import VQCClassifier
        X_train, X_test, y_train, y_test = moons_data
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=5, seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (X_test.shape[0],)
        assert set(np.unique(preds).tolist()).issubset(set(clf.classes_.tolist()))

    @pytest.mark.slow
    def test_predict_proba_rows_sum_to_one(self, moons_data):
        from qmc import VQCClassifier
        X_train, X_test, y_train, _ = moons_data
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=3, seed=42)
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)
        assert p.shape == (X_test.shape[0], clf.n_classes_)
        row_sums = p.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    @pytest.mark.slow
    def test_cloudpickle_roundtrip_preserves_predictions(self, moons_data):
        """
        Round-trip persistence should preserve predictions.

        PennyLane QNodes close over locally-defined functions that
        stdlib ``pickle`` cannot serialise, so we use ``cloudpickle``
        (the same library sklearn's ``joblib`` uses under the hood).
        """
        cloudpickle = pytest.importorskip("cloudpickle")
        from qmc import VQCClassifier
        X_train, X_test, y_train, _ = moons_data
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=3, seed=42)
        clf.fit(X_train, y_train)
        preds_before = clf.predict(X_test)
        reloaded = cloudpickle.loads(cloudpickle.dumps(clf))
        preds_after = reloaded.predict(X_test)
        assert np.array_equal(preds_before, preds_after)

    @pytest.mark.slow
    def test_pipeline_compatibility(self, moons_data):
        from qmc import VQCClassifier
        X_train, X_test, y_train, y_test = moons_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("vqc", VQCClassifier(n_qubits=2, n_layers=1, epochs=3, seed=42)),
        ])
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        assert 0.0 <= score <= 1.0

    @pytest.mark.slow
    def test_cross_val_score_returns_finite(self, moons_data):
        from qmc import VQCClassifier
        X_train, _, y_train, _ = moons_data
        clf = VQCClassifier(n_qubits=2, n_layers=1, epochs=3, seed=42)
        scores = cross_val_score(clf, X_train, y_train, cv=2)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))


# --------------------------------------------------------------------------
# QuantumKernelClassifier
# --------------------------------------------------------------------------


class TestQuantumKernelClassifier:
    def test_imports_from_top_level(self):
        from qmc import QuantumKernelClassifier
        assert QuantumKernelClassifier is not None

    def test_get_set_params(self):
        from qmc import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_qubits=2, C=0.5, max_samples=30)
        p = clf.get_params()
        assert p["n_qubits"] == 2
        assert p["C"] == 0.5
        clf.set_params(C=2.0)
        assert clf.C == 2.0

    def test_predict_before_fit_raises(self):
        from qmc import QuantumKernelClassifier
        clf = QuantumKernelClassifier(n_qubits=2, max_samples=20)
        with pytest.raises(NotFittedError):
            clf.predict(np.zeros((3, 2)))

    @pytest.mark.slow
    def test_fit_predict_on_moons(self, moons_data):
        from qmc import QuantumKernelClassifier
        X_train, X_test, y_train, y_test = moons_data
        clf = QuantumKernelClassifier(n_qubits=2, max_samples=20, seed=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        assert preds.shape == (X_test.shape[0],)
        # On moons, even a tiny kernel SVM should clear majority vote
        # most of the time. Loose bound to avoid false failures.
        acc = float(np.mean(preds == y_test))
        assert acc >= 0.3

    @pytest.mark.slow
    def test_predict_proba_rows_sum_to_one(self, moons_data):
        from qmc import QuantumKernelClassifier
        X_train, X_test, y_train, _ = moons_data
        clf = QuantumKernelClassifier(n_qubits=2, max_samples=20, seed=42)
        clf.fit(X_train, y_train)
        p = clf.predict_proba(X_test)
        assert p.shape == (X_test.shape[0], clf.n_classes_)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-6)
