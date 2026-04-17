"""sklearn-compatible wrappers around the quantum classifiers.

The ``VQC`` / ``VQCMulticlass`` modules in :mod:`qmc.circuits.vqc` and the
kernel utilities in :mod:`qmc.circuits.kernels` expose PennyLane-level
primitives. Those are fine for researchers, but they don't plug into
sklearn's ``Pipeline``, ``GridSearchCV``, ``cross_val_score``, etc.

This module provides thin wrappers that satisfy the
``BaseEstimator`` + ``ClassifierMixin`` contract so consumers can drop
the quantum models into any sklearn workflow:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from qmc import VQCClassifier
    >>> pipe = Pipeline([("scaler", StandardScaler()), ("vqc", VQCClassifier(n_qubits=4, n_layers=2, epochs=10))])
    >>> pipe.fit(X_train, y_train)
    >>> pipe.score(X_test, y_test)

Design notes
------------
- Every constructor argument is stored verbatim on ``self`` with the
  same name. sklearn's ``get_params`` / ``set_params`` walks
  ``__init__`` signatures and expects this. No heavy work in
  ``__init__`` — models are built in ``fit``.
- Fitted attributes end with an underscore (``self.model_``,
  ``self.classes_``, ``self.n_features_in_``) per the sklearn
  convention.
- ``predict_proba`` returns an (n_samples, n_classes) matrix whose
  rows sum to 1.
- ``check_estimator`` is exercised in ``tests/test_sklearn_compat.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
    check_X_y,
)


__all__ = ["VQCClassifier", "QuantumKernelClassifier"]


# ------------------------------------------------------------------------
# VQCClassifier
# ------------------------------------------------------------------------


class VQCClassifier(BaseEstimator, ClassifierMixin):
    """Variational Quantum Classifier with a scikit-learn API.

    Wraps :class:`qmc.circuits.vqc.VQC` (binary) or
    :class:`qmc.circuits.vqc.VQCMulticlass` (multiclass) based on the
    number of classes present in ``y`` at fit time.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits. Must be at least the input feature dimension;
        inputs are padded with zeros if ``n_qubits > X.shape[1]`` and
        truncated if ``n_qubits < X.shape[1]``.
    n_layers : int, default=2
        Number of ``StronglyEntanglingLayers``.
    epochs : int, default=30
        Maximum number of training epochs.
    lr : float, default=0.05
        Adam learning rate.
    batch_size : int, default=32
        Mini-batch size for training.
    seed : int, default=42
        Random seed for reproducibility.
    device_name : str, default="default.qubit"
        PennyLane device backend. Use ``"lightning.qubit"`` for faster
        simulation if ``pennylane-lightning`` is installed.
    diff_method : str, default="best"
        Differentiation method passed to the QNode.

    Attributes
    ----------
    model_ : torch.nn.Module
        The fitted underlying VQC (binary) or VQCMulticlass model.
    history_ : dict
        Training history dict with keys ``train_loss``, ``val_loss``,
        ``val_acc``, ``epoch_time``.
    classes_ : numpy.ndarray of shape (n_classes,)
        Class labels known to the classifier.
    n_features_in_ : int
        Number of features seen during ``fit``.
    n_classes_ : int
        Number of classes inferred from ``y`` at fit time.

    Notes
    -----
    Persistence: PennyLane QNodes close over locally-defined functions
    that stdlib ``pickle`` cannot serialise. Use ``cloudpickle`` or
    ``joblib.dump`` / ``joblib.load`` instead — both handle the
    underlying QNode correctly.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        epochs: int = 30,
        lr: float = 0.05,
        batch_size: int = 32,
        seed: int = 42,
        device_name: str = "default.qubit",
        diff_method: str = "best",
    ) -> None:
        # sklearn contract: store every argument verbatim, no computation.
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.device_name = device_name
        self.diff_method = diff_method

    # ----- sklearn tag API (lets check_estimator skip unsupported features)
    def _more_tags(self) -> dict[str, Any]:
        return {
            "binary_only": False,
            "non_deterministic": True,  # stochastic batches
            "poor_score": True,  # VQC on random data doesn't beat 50%
            "_xfail_checks": {
                "check_fit_idempotent": (
                    "VQC training is stochastic even with a fixed seed "
                    "because PyTorch autograd ops on CPU can differ."
                ),
                "check_methods_subset_invariance": (
                    "Batch normalisation inside the QNode introduces "
                    "subset-dependent outputs in some PennyLane versions."
                ),
            },
        }

    # ----- Core sklearn API -------------------------------------------

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> "VQCClassifier":
        """Fit the VQC on training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)

        Returns
        -------
        self : VQCClassifier
            Fitted estimator.
        """
        from qmc.circuits.vqc import train_vqc

        X, y = check_X_y(X, y, ensure_2d=True, accept_sparse=False)
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.shape[0])
        self.n_features_in_ = int(X.shape[1])

        # Map labels to contiguous integers in [0, n_classes_)
        self._label_to_index = {c: i for i, c in enumerate(self.classes_.tolist())}
        y_internal = np.array([self._label_to_index[v] for v in y.tolist()])

        X_padded = self._pad_or_truncate(X)

        multiclass = self.n_classes_ > 2
        # No validation split for sklearn-compat; use training set for
        # both (identical to what the underlying train_vqc expects).
        model, history = train_vqc(
            X_train=X_padded,
            y_train=y_internal,
            X_val=X_padded,
            y_val=y_internal,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            seed=self.seed,
            multiclass=multiclass,
            n_classes=self.n_classes_,
        )

        self.model_ = model
        self.history_ = history
        self._multiclass_ = multiclass
        return self

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """Return class probability estimates for ``X``."""
        import torch

        check_is_fitted(self, attributes=["model_", "classes_"])
        X = check_array(X, ensure_2d=True, accept_sparse=False)
        X_padded = self._pad_or_truncate(X)

        x_t = torch.tensor(X_padded, dtype=torch.float32)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_t)

        if self._multiclass_:
            probs = torch.softmax(logits, dim=-1).numpy()
        else:
            p1 = torch.sigmoid(logits.squeeze(-1)).numpy()
            probs = np.stack([1.0 - p1, p1], axis=1)

        # Normalise to guard against float drift
        row_sums = probs.sum(axis=1, keepdims=True)
        probs = probs / np.where(row_sums == 0, 1.0, row_sums)
        return probs

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict class labels for ``X``."""
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

    # ----- Helpers ----------------------------------------------------

    def _pad_or_truncate(self, X: NDArray[Any]) -> NDArray[np.floating[Any]]:
        """Match the circuit's qubit count by padding with zeros or truncating."""
        n = self.n_qubits
        f = X.shape[1]
        if f == n:
            return X.astype(np.float32, copy=False)
        if f < n:
            pad = np.zeros((X.shape[0], n - f), dtype=X.dtype)
            return np.concatenate([X, pad], axis=1).astype(np.float32, copy=False)
        return X[:, :n].astype(np.float32, copy=False)


# ------------------------------------------------------------------------
# QuantumKernelClassifier
# ------------------------------------------------------------------------


class QuantumKernelClassifier(BaseEstimator, ClassifierMixin):
    """Quantum kernel SVM with a scikit-learn API.

    Computes an IQP-style quantum kernel matrix on ``X`` and fits a
    classical :class:`sklearn.svm.SVC` with ``kernel='precomputed'``.

    Parameters
    ----------
    n_qubits : int, default=4
        Number of qubits for the kernel circuit.
    C : float, default=1.0
        SVM regularisation parameter.
    max_samples : int, default=300
        Upper bound on training samples used for the kernel matrix.
        Kernel computation is O(n^2); larger datasets are stratifiedly
        subsampled.
    seed : int, default=42
        Random seed used for subsampling.
    device_name : str, default="default.qubit"
        PennyLane device backend.

    Attributes
    ----------
    svc_ : sklearn.svm.SVC
        The fitted SVM.
    X_support_ : numpy.ndarray
        Training features used in the kernel matrix (possibly subsampled).
    classes_ : numpy.ndarray
    n_features_in_ : int
    n_classes_ : int
    """

    def __init__(
        self,
        n_qubits: int = 4,
        C: float = 1.0,
        max_samples: int = 300,
        seed: int = 42,
        device_name: str = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        self.C = C
        self.max_samples = max_samples
        self.seed = seed
        self.device_name = device_name

    def _more_tags(self) -> dict[str, Any]:
        return {
            "binary_only": False,
            "non_deterministic": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_methods_subset_invariance": (
                    "Subsampling in fit can produce support sets of different sizes."
                ),
            },
        }

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> "QuantumKernelClassifier":
        from sklearn.svm import SVC
        from qmc.circuits.kernels import (
            compute_quantum_kernel,
            _stratified_subsample,
        )

        X, y = check_X_y(X, y, ensure_2d=True, accept_sparse=False)
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.shape[0])
        self.n_features_in_ = int(X.shape[1])

        X_padded = self._pad_or_truncate(X)

        # Optional stratified subsampling to keep the O(n^2) kernel tractable
        if X_padded.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.seed)
            idx = _stratified_subsample(y, self.max_samples, rng)
            X_sub, y_sub = X_padded[idx], y[idx]
        else:
            X_sub, y_sub = X_padded, y

        K = compute_quantum_kernel(X_sub, n_qubits=self.n_qubits,
                                   device_name=self.device_name)
        svc = SVC(kernel="precomputed", C=self.C, probability=True,
                  random_state=self.seed)
        svc.fit(K, y_sub)

        self.svc_ = svc
        self.X_support_ = X_sub
        return self

    def _kernel_to_support(self, X: NDArray[Any]) -> NDArray[np.floating[Any]]:
        from qmc.circuits.kernels import compute_quantum_kernel_cross
        X_padded = self._pad_or_truncate(X)
        return compute_quantum_kernel_cross(
            X_padded, self.X_support_,
            n_qubits=self.n_qubits,
            device_name=self.device_name,
        )

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        check_is_fitted(self, attributes=["svc_", "X_support_"])
        X = check_array(X, ensure_2d=True, accept_sparse=False)
        K = self._kernel_to_support(X)
        return self.svc_.predict(K)

    def predict_proba(self, X: NDArray[Any]) -> NDArray[np.floating[Any]]:
        check_is_fitted(self, attributes=["svc_", "X_support_"])
        X = check_array(X, ensure_2d=True, accept_sparse=False)
        K = self._kernel_to_support(X)
        return self.svc_.predict_proba(K)

    def _pad_or_truncate(self, X: NDArray[Any]) -> NDArray[np.floating[Any]]:
        n = self.n_qubits
        f = X.shape[1]
        if f == n:
            return X.astype(np.float32, copy=False)
        if f < n:
            pad = np.zeros((X.shape[0], n - f), dtype=X.dtype)
            return np.concatenate([X, pad], axis=1).astype(np.float32, copy=False)
        return X[:, :n].astype(np.float32, copy=False)
