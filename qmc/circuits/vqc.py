"""
Variational Quantum Classifier (VQC) for classification tasks.

Architecture
------------
- Configurable qubits (default: 8)
- AngleEmbedding (RX rotations)
- StronglyEntanglingLayers (configurable depth)
- Single-qubit readout (binary) or all-qubit readout (multiclass)

The QNode is wrapped with ``qml.qnn.TorchLayer`` so the whole model
is a standard ``nn.Module`` trainable with PyTorch optimizers.
"""

import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pennylane as qml

from qmc.circuits.templates import (
    angle_encoding_circuit,
    multi_output_circuit,
    get_weight_shapes,
)


# ------------------------------------------------------------------
# Binary VQC
# ------------------------------------------------------------------

class VQC(nn.Module):
    """
    Binary classifier backed by a variational quantum circuit.

    Uses a single-qubit PauliZ readout suitable for binary classification
    with BCEWithLogitsLoss.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match input feature dimension). Default: 8.
    n_layers : int
        Number of StronglyEntanglingLayers. Default: 4.
    device_name : str
        PennyLane device backend. Default: 'default.qubit'.
    diff_method : str
        Differentiation method. Default: 'best'.
    """

    def __init__(self, n_qubits=8, n_layers=4, device_name="default.qubit",
                 diff_method="best"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        qnode = angle_encoding_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            diff_method=diff_method,
        )
        weight_shapes = get_weight_shapes(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_qubits)

        Returns
        -------
        torch.Tensor, shape (batch,)
            Raw logits (apply sigmoid externally or use BCEWithLogitsLoss).
        """
        return self.qlayer(x)


# ------------------------------------------------------------------
# Multiclass VQC
# ------------------------------------------------------------------

class VQCMulticlass(nn.Module):
    """
    Multi-class classifier backed by a variational quantum circuit.

    Measures PauliZ on all qubits and feeds the resulting feature vector
    through a classical linear readout head.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must match input feature dimension). Default: 8.
    n_layers : int
        Number of StronglyEntanglingLayers. Default: 4.
    n_classes : int
        Number of output classes. Default: 2.
    device_name : str
        PennyLane device backend. Default: 'default.qubit'.
    diff_method : str
        Differentiation method. Default: 'best'.
    """

    def __init__(self, n_qubits=8, n_layers=4, n_classes=2,
                 device_name="default.qubit", diff_method="best"):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes

        qnode = multi_output_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name,
            diff_method=diff_method,
        )
        weight_shapes = get_weight_shapes(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # Classical readout head: map n_qubits quantum features to n_classes
        self.classifier = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_qubits)

        Returns
        -------
        torch.Tensor, shape (batch, n_classes)
            Raw logits (use CrossEntropyLoss).
        """
        q_features = self.qlayer(x)
        return self.classifier(q_features)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_vqc(
    X_train,
    y_train,
    X_val,
    y_val,
    n_qubits=8,
    n_layers=4,
    epochs=50,
    lr=0.01,
    batch_size=32,
    seed=42,
    multiclass=False,
    n_classes=2,
):
    """
    Train a VQC classifier.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training features and labels.
    X_val, y_val : np.ndarray
        Validation features and labels.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of entangling layers.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Mini-batch size.
    seed : int
        Random seed for reproducibility.
    multiclass : bool
        If True, use VQCMulticlass with CrossEntropyLoss.
        If False, use VQC with BCEWithLogitsLoss.
    n_classes : int
        Number of classes (only used when multiclass=True).

    Returns
    -------
    model : VQC or VQCMulticlass
        Best model (by validation loss).
    history : dict
        Keys: train_loss, val_loss, val_acc, epoch_time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # quantum sim is CPU-only

    # Tensors
    X_tr = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_v = torch.tensor(X_val, dtype=torch.float32, device=device)

    if multiclass:
        y_tr = torch.tensor(y_train, dtype=torch.long, device=device)
        y_v = torch.tensor(y_val, dtype=torch.long, device=device)
        model = VQCMulticlass(n_qubits=n_qubits, n_layers=n_layers,
                              n_classes=n_classes).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device)
        model = VQC(n_qubits=n_qubits, n_layers=n_layers).to(device)
        criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": [],
    }
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # --- train ---
        model.train()
        running_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            if multiclass:
                loss = criterion(logits, yb)
            else:
                loss = criterion(logits.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            if multiclass:
                val_loss = criterion(val_logits, y_v).item()
                val_preds = val_logits.argmax(dim=-1)
                val_acc = (val_preds == y_v).float().mean().item()
            else:
                val_logits_sq = val_logits.squeeze(-1)
                val_loss = criterion(val_logits_sq, y_v).item()
                val_preds = (torch.sigmoid(val_logits_sq) >= 0.5).float()
                val_acc = (val_preds == y_v).float().mean().item()

        elapsed = time.time() - t0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(elapsed)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[VQC] Epoch {epoch:>3d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
