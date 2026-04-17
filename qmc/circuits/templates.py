"""
Reusable quantum circuit templates for QML experiments.

Provides parameterized circuits for:
  - Binary classification (single-qubit readout)
  - Multi-class classification (all-qubit readout)
  - Quantum kernel estimation (fidelity-based)
"""

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np


# ------------------------------------------------------------------
# Device factory
# ------------------------------------------------------------------

def create_device(n_qubits, device_name="default.qubit"):
    """
    Create a PennyLane device.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    device_name : str
        PennyLane device backend (default: 'default.qubit').

    Returns
    -------
    qml.Device
    """
    return qml.device(device_name, wires=n_qubits)


# ------------------------------------------------------------------
# Binary VQC circuit (single expval readout)
# ------------------------------------------------------------------

def angle_encoding_circuit(n_qubits=8, n_layers=4,
                           device_name="default.qubit",
                           diff_method="best"):
    """
    Return a QNode for a variational quantum classifier.

    Encoding : AngleEmbedding (RX rotations)
    Ansatz   : StronglyEntanglingLayers
    Readout  : expval(PauliZ(0))

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 8).
    n_layers : int
        Number of entangling layers (default: 4).
    device_name : str
        PennyLane device backend.
    diff_method : str
        Differentiation method for the QNode.

    Returns
    -------
    qml.QNode
    """
    dev = create_device(n_qubits, device_name)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


# ------------------------------------------------------------------
# Multi-output circuit (all-qubit expval readout)
# ------------------------------------------------------------------

def multi_output_circuit(n_qubits=8, n_layers=4,
                         device_name="default.qubit",
                         diff_method="best"):
    """
    Return a QNode that measures expval(PauliZ) on every qubit.

    Useful for multi-class classification where each qubit contributes
    a continuous feature to a classical readout head.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 8).
    n_layers : int
        Number of entangling layers (default: 4).
    device_name : str
        PennyLane device backend.
    diff_method : str
        Differentiation method for the QNode.

    Returns
    -------
    qml.QNode
    """
    dev = create_device(n_qubits, device_name)

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ------------------------------------------------------------------
# Quantum kernel circuit (fidelity estimation)
# ------------------------------------------------------------------

def kernel_circuit(n_qubits=8, device_name="default.qubit"):
    """
    Return a QNode for quantum kernel estimation via IQP-style encoding.

    The kernel value is the squared overlap |<0|U(x2)^dag U(x1)|0>|^2.
    Each data vector is encoded with AngleEmbedding followed by a layer
    of entangling IQPEmbedding (ZZ interactions).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 8).
    device_name : str
        PennyLane device backend.

    Returns
    -------
    qml.QNode
    """
    dev = create_device(n_qubits, device_name)

    @qml.qnode(dev, interface="autograd")
    def _kernel(x1, x2):
        # Encode x1
        qml.AngleEmbedding(x1, wires=range(n_qubits), rotation="X")
        qml.IQPEmbedding(x1, wires=range(n_qubits), n_repeats=1)
        # Adjoint of x2 encoding
        qml.adjoint(qml.IQPEmbedding)(x2, wires=range(n_qubits), n_repeats=1)
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits), rotation="X")
        return qml.probs(wires=range(n_qubits))

    return _kernel


# ------------------------------------------------------------------
# Weight shape helper
# ------------------------------------------------------------------

def get_weight_shapes(n_qubits=8, n_layers=4):
    """
    Return the weight_shapes dict expected by qml.qnn.TorchLayer.

    For StronglyEntanglingLayers the weight tensor has shape
    (n_layers, n_qubits, 3).

    Parameters
    ----------
    n_qubits : int
        Number of qubits (default: 8).
    n_layers : int
        Number of entangling layers (default: 4).

    Returns
    -------
    dict[str, tuple[int, ...]]
    """
    return {"weights": (n_layers, n_qubits, 3)}
