"""
Quantum kernel methods for classification.

Uses IQP-style feature-map encoding (AngleEmbedding + IQPEmbedding)
to compute a quantum kernel matrix, then trains a classical SVM on top.

Because the kernel matrix is O(n^2), training data is sub-sampled to
``max_samples`` (default 300) when the dataset is large.
"""

import time

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pennylane import numpy as pnp

from qmc.circuits.templates import kernel_circuit as _build_kernel_circuit


# ------------------------------------------------------------------
# Kernel computation
# ------------------------------------------------------------------

def _kernel_value(x1, x2, kernel_qnode):
    """
    Compute a single kernel entry k(x1, x2) = |<0|U(x2)^dag U(x1)|0>|^2.

    The circuit returns the full probability vector; the kernel value is
    the probability of the all-zero state (index 0).
    """
    probs = kernel_qnode(x1, x2)
    return float(probs[0])


def compute_quantum_kernel(X, n_qubits=8, device_name="default.qubit"):
    """
    Compute the quantum kernel (Gram) matrix for dataset X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix. n_features must equal n_qubits.
    n_qubits : int
        Number of qubits (default: 8).
    device_name : str
        PennyLane device backend.

    Returns
    -------
    K : np.ndarray, shape (n_samples, n_samples)
        Symmetric positive-semidefinite kernel matrix.
    """
    assert X.shape[1] == n_qubits, (
        f"Feature dim ({X.shape[1]}) must match n_qubits ({n_qubits})."
    )

    kernel_qnode = _build_kernel_circuit(n_qubits=n_qubits, device_name=device_name)

    n = X.shape[0]
    K = np.zeros((n, n))
    X_pnp = pnp.array(X, requires_grad=False)

    total_pairs = n * (n + 1) // 2
    computed = 0
    t0 = time.time()

    for i in range(n):
        for j in range(i, n):
            k_ij = _kernel_value(X_pnp[i], X_pnp[j], kernel_qnode)
            K[i, j] = k_ij
            K[j, i] = k_ij
            computed += 1

        if (i + 1) % 50 == 0 or i == n - 1:
            elapsed = time.time() - t0
            print(
                f"  [Kernel] row {i+1}/{n} | "
                f"{computed}/{total_pairs} pairs | "
                f"{elapsed:.1f}s elapsed"
            )

    return K


def compute_quantum_kernel_cross(X1, X2, n_qubits=8,
                                  device_name="default.qubit"):
    """
    Compute the cross kernel matrix K(X1, X2).

    Parameters
    ----------
    X1 : np.ndarray, shape (n1, n_features)
        First feature matrix.
    X2 : np.ndarray, shape (n2, n_features)
        Second feature matrix.
    n_qubits : int
        Number of qubits (default: 8).
    device_name : str
        PennyLane device backend.

    Returns
    -------
    K : np.ndarray, shape (n1, n2)
    """
    assert X1.shape[1] == n_qubits and X2.shape[1] == n_qubits

    kernel_qnode = _build_kernel_circuit(n_qubits=n_qubits, device_name=device_name)

    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    X1_pnp = pnp.array(X1, requires_grad=False)
    X2_pnp = pnp.array(X2, requires_grad=False)

    for i in range(n1):
        for j in range(n2):
            K[i, j] = _kernel_value(X1_pnp[i], X2_pnp[j], kernel_qnode)

    return K


# ------------------------------------------------------------------
# Train & evaluate
# ------------------------------------------------------------------

def train_quantum_kernel_svm(
    X_train,
    y_train,
    X_test,
    y_test,
    n_qubits=8,
    max_samples=300,
    seed=42,
):
    """
    Train an SVM with a quantum kernel and return evaluation metrics.

    If the training set exceeds ``max_samples``, a stratified subsample
    is drawn.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data.
    n_qubits : int
        Number of qubits (default: 8).
    max_samples : int
        Maximum number of samples for kernel computation (default: 300).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    metrics : dict
        Keys: accuracy, f1, roc_auc, train_time,
        kernel_time_train, kernel_time_test, n_train, n_test.
    svm : SVC
        Fitted SVM model.
    """
    rng = np.random.RandomState(seed)

    # Subsample if necessary
    if len(X_train) > max_samples:
        idx = _stratified_subsample(y_train, max_samples, rng)
        X_train = X_train[idx]
        y_train = y_train[idx]
        print(f"  [Kernel SVM] Subsampled training set to {max_samples} samples.")

    if len(X_test) > max_samples:
        idx = _stratified_subsample(y_test, max_samples, rng)
        X_test = X_test[idx]
        y_test = y_test[idx]
        print(f"  [Kernel SVM] Subsampled test set to {max_samples} samples.")

    # --- Compute kernel matrices ---
    print(f"  [Kernel SVM] Computing train kernel ({len(X_train)}x{len(X_train)})...")
    t0 = time.time()
    K_train = compute_quantum_kernel(X_train, n_qubits=n_qubits)
    kernel_time_train = time.time() - t0

    print(f"  [Kernel SVM] Computing test kernel ({len(X_test)}x{len(X_train)})...")
    t1 = time.time()
    K_test = compute_quantum_kernel_cross(X_test, X_train, n_qubits=n_qubits)
    kernel_time_test = time.time() - t1

    # --- Train SVM ---
    t2 = time.time()
    svm = SVC(kernel="precomputed", probability=True, random_state=seed)
    svm.fit(K_train, y_train)
    train_time = time.time() - t2

    # --- Evaluate ---
    y_pred = svm.predict(K_test)
    y_proba = svm.predict_proba(K_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="weighted"
            )
    except ValueError:
        auc = float("nan")

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": auc,
        "train_time": train_time,
        "kernel_time_train": kernel_time_train,
        "kernel_time_test": kernel_time_test,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    print(
        f"  [Kernel SVM] acc={acc:.4f} | f1={f1:.4f} | "
        f"auc={auc:.4f} | kernel_train={kernel_time_train:.1f}s"
    )

    return metrics, svm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _stratified_subsample(y, n_samples, rng):
    """Return indices for a stratified subsample of size n_samples."""
    classes, counts = np.unique(y, return_counts=True)
    indices = []
    for cls, cnt in zip(classes, counts):
        cls_idx = np.where(y == cls)[0]
        n_take = max(1, int(round(n_samples * cnt / len(y))))
        n_take = min(n_take, len(cls_idx))
        indices.extend(rng.choice(cls_idx, size=n_take, replace=False).tolist())

    rng.shuffle(indices)
    return np.array(indices[:n_samples])
