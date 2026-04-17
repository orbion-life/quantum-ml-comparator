"""
Classical ML Models
===================
Classical models for fair comparison with quantum circuits.

Includes:
  - TinyMLP: Minimal neural network
  - MediumMLP: Moderately sized neural network
  - Scikit-learn wrappers: SVM, Random Forest, Logistic Regression
  - Training and evaluation utilities
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ===================================================================
# PyTorch Models
# ===================================================================

class TinyMLP(nn.Module):
    """
    Minimal MLP for parameter-matched comparison with quantum models.

    Default architecture: input_dim -> 12 -> num_classes (~122 params for
    8-dim input, binary output).

    Parameters
    ----------
    input_dim : int
        Input feature dimension (default: 8).
    num_classes : int
        Number of output classes (default: 2).
    hidden : list of int or None
        Hidden layer sizes. Default: [12].
    """

    def __init__(self, input_dim=8, num_classes=2, hidden=None):
        super().__init__()
        h = hidden or [12]
        layers = []
        prev = input_dim
        for dim in h:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MediumMLP(nn.Module):
    """
    Medium-sized MLP for comparison with quantum models.

    Default architecture: input_dim -> 32 -> 16 -> num_classes (~578 params
    for 8-dim input, binary output).

    Parameters
    ----------
    input_dim : int
        Input feature dimension (default: 8).
    num_classes : int
        Number of output classes (default: 2).
    hidden : list of int or None
        Hidden layer sizes. Default: [32, 16].
    """

    def __init__(self, input_dim=8, num_classes=2, hidden=None):
        super().__init__()
        h = hidden or [32, 16]
        layers = []
        prev = input_dim
        for dim in h:
            layers.append(nn.Linear(prev, dim))
            layers.append(nn.ReLU())
            prev = dim
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===================================================================
# Scikit-learn wrappers
# ===================================================================

def get_svm(C=1.0, gamma="scale", kernel="rbf", seed=42):
    """
    Create an SVM classifier with RBF kernel.

    Parameters
    ----------
    C : float
        Regularization parameter (default: 1.0).
    gamma : str or float
        Kernel coefficient (default: 'scale').
    kernel : str
        Kernel type (default: 'rbf').
    seed : int
        Random seed (default: 42).

    Returns
    -------
    sklearn.svm.SVC
        Unfitted SVM classifier with probability estimates enabled.
    """
    return SVC(
        C=C,
        gamma=gamma,
        kernel=kernel,
        probability=True,
        random_state=seed,
    )


def get_random_forest(n_estimators=100, seed=42):
    """
    Create a Random Forest classifier.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default: 100).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Unfitted Random Forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )


def get_logistic_regression(C=1.0, max_iter=1000, seed=42):
    """
    Create a Logistic Regression classifier.

    Parameters
    ----------
    C : float
        Inverse of regularization strength (default: 1.0).
    max_iter : int
        Maximum number of iterations (default: 1000).
    seed : int
        Random seed (default: 42).

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Unfitted Logistic Regression classifier.
    """
    # Note: `multi_class` was removed in sklearn 1.7. The behavior previously
    # selected by `multi_class="auto"` is now the default and only supported
    # path, so we omit the argument.
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )


# ===================================================================
# Training / evaluation utilities
# ===================================================================

def _safe_auc(y_true, y_score, **kwargs):
    """AUC-ROC that returns 0.0 when only one class is present."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    try:
        return roc_auc_score(y_true, y_score, **kwargs)
    except ValueError:
        return 0.0


def train_pytorch_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=200,
    lr=0.001,
    batch_size=64,
    patience_limit=30,
    verbose=True,
):
    """
    Standard PyTorch training loop with early-stopping on validation loss.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to train.
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray
        Validation data and labels.
    epochs : int
        Maximum number of training epochs (default: 200).
    lr : float
        Learning rate (default: 0.001).
    batch_size : int
        Mini-batch size (default: 64).
    patience_limit : int
        Early stopping patience (default: 30).
    verbose : bool
        Whether to print progress (default: True).

    Returns
    -------
    model : nn.Module
        Best checkpoint restored.
    history : dict
        Keys: train_loss, val_loss.
    """
    device = next(model.parameters()).device

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.long).to(device)
    X_v = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_v = torch.tensor(y_val, dtype=torch.long).to(device)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_tr)
        history["train_loss"].append(epoch_loss)

        # -- validate --
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = criterion(val_logits, y_v).item()
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and epoch % 50 == 0:
            print(
                f"  Epoch {epoch:4d}/{epochs}  "
                f"train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}"
            )

        if patience_counter >= patience_limit:
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, X_test, y_test, model_name=""):
    """
    Compute F1 (macro/weighted), precision, recall, AUC-ROC.

    Works with both PyTorch nn.Module and scikit-learn estimators.

    Parameters
    ----------
    model : nn.Module or sklearn estimator
        Trained model.
    X_test : np.ndarray, shape (N, D)
        Test features.
    y_test : np.ndarray, shape (N,)
        Test labels.
    model_name : str
        Label for printed output (default: empty).

    Returns
    -------
    dict
        Metric name -> float.
    """
    is_torch = isinstance(model, nn.Module)

    if is_torch:
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = model(X_t).cpu().numpy()
        y_prob = _softmax(logits)
        y_pred = logits.argmax(axis=1)
    else:
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None

    num_classes = len(np.unique(y_test))

    metrics = {}
    for avg in ["macro", "weighted"]:
        metrics[f"f1_{avg}"] = f1_score(
            y_test, y_pred, average=avg, zero_division=0
        )
        metrics[f"precision_{avg}"] = precision_score(
            y_test, y_pred, average=avg, zero_division=0
        )
        metrics[f"recall_{avg}"] = recall_score(
            y_test, y_pred, average=avg, zero_division=0
        )

    # Per-class F1
    per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, f in enumerate(per_class):
        metrics[f"f1_class_{i}"] = float(f)

    # AUC-ROC
    if y_prob is not None:
        if num_classes == 2:
            prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics["auc_roc"] = _safe_auc(y_test, prob_pos)
        else:
            metrics["auc_roc"] = _safe_auc(
                y_test, y_prob, multi_class="ovr", average="macro"
            )

    metrics["accuracy"] = float(np.mean(y_test == y_pred))

    if model_name:
        print(
            f"\n  [{model_name}]  "
            f"F1-macro={metrics['f1_macro']:.4f}  "
            f"F1-weighted={metrics['f1_weighted']:.4f}  "
            f"Acc={metrics['accuracy']:.4f}"
            + (
                f"  AUC={metrics.get('auc_roc', 0):.4f}"
                if "auc_roc" in metrics
                else ""
            )
        )

    return metrics


def _softmax(logits):
    """Numerically stable softmax over last axis."""
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def count_params(model):
    """
    Count total trainable parameters in a PyTorch module.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.

    Returns
    -------
    int
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
