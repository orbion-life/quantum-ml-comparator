"""
Quantum Number Preserving (QNP) Gates for VQE
==============================================
Implementation of Anselmetti et al. 2021 (New J. Phys. 23, 113010)

These gates preserve:
  - N_alpha (number of spin-up electrons)
  - N_beta (number of spin-down electrons)
  - S^2 (total spin squared)

Gate fabric for F(2^{2M}) in Jordan-Wigner representation with
interleaved spin ordering: ...1beta 1alpha 0beta 0alpha

Two primitive gates:
  QNPOR(phi): Spatial orbital rotation (4 CNOTs)
  QNPPX(theta): Diagonal pair exchange (13 CNOTs)

Combined into Q = QNPPX(theta) * QNPOR(phi) per spatial orbital pair.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


# ============================================================
# QNPOR: Spatial Orbital Rotation Gate
# ============================================================

def qnp_orbital_rotation(phi, wires):
    """
    Apply QNPOR(phi) gate to 4 qubits in interleaved ordering.

    Implements a spin-adapted Givens rotation: the same orbital rotation
    is applied to both the alpha and beta spin channels.

    Parameters
    ----------
    phi : float
        Orbital rotation angle.
    wires : list of 4 ints
        [p_alpha, p_beta, q_alpha, q_beta] qubit indices.
    """
    p_a, p_b, q_a, q_b = wires

    # Alpha channel: Givens rotation on qubits p_alpha, q_alpha
    qml.CNOT(wires=[q_a, p_a])
    qml.RY(phi, wires=p_a)
    qml.CNOT(wires=[p_a, q_a])
    qml.RY(-phi, wires=p_a)
    qml.CNOT(wires=[q_a, p_a])

    # Beta channel: Same Givens rotation on qubits p_beta, q_beta
    qml.CNOT(wires=[q_b, p_b])
    qml.RY(phi, wires=p_b)
    qml.CNOT(wires=[p_b, q_b])
    qml.RY(-phi, wires=p_b)
    qml.CNOT(wires=[q_b, p_b])


# ============================================================
# QNPPX: Diagonal Pair Exchange Gate
# ============================================================

def qnp_pair_exchange(theta, wires):
    """
    Apply QNPPX(theta) gate to 4 qubits in interleaved ordering.

    Implements a pair exchange: simultaneously moves one alpha and one beta
    electron from orbitals (q_alpha, q_beta) to (p_alpha, p_beta), preserving
    all quantum numbers.

    Parameters
    ----------
    theta : float
        Pair exchange angle.
    wires : list of 4 ints
        [p_alpha, p_beta, q_alpha, q_beta] qubit indices.
    """
    p_a, p_b, q_a, q_b = wires

    # Equivalent to exp(-i*theta*(a_dag_pa * a_dag_pb * a_qb * a_qa - h.c.))
    # PennyLane DoubleExcitation uses theta/2 convention
    qml.DoubleExcitation(2 * theta, wires=[q_a, q_b, p_b, p_a])


# ============================================================
# Combined QNP Gate Element
# ============================================================

def qnp_gate(phi, theta, wires, pi_gate=True):
    """
    Apply the full Q gate = QNPPX(theta) * QNPOR(phi) [* Pi] to 4 qubits.

    Parameters
    ----------
    phi : float
        Orbital rotation angle.
    theta : float
        Pair exchange angle.
    wires : list of 4 ints
        [p_alpha, p_beta, q_alpha, q_beta] qubit indices.
    pi_gate : bool
        If True, prepend Pi = QNPOR(pi) for better trainability
        (Strategy A from Anselmetti -- avoids barren plateaus).
    """
    if pi_gate:
        qnp_orbital_rotation(np.pi, wires)

    qnp_orbital_rotation(phi, wires)
    qnp_pair_exchange(theta, wires)


# ============================================================
# QNP Gate Fabric
# ============================================================

def qnp_fabric(params, n_qubits, n_layers, pi_gate=True):
    """
    Apply the full QNP gate fabric.

    The fabric is a tessellation of Q gates over spatial orbital pairs
    in alternating even/odd layers.

    Parameters
    ----------
    params : array of shape (n_layers, n_gates_per_layer, 2)
        params[l, g, 0] = phi (orbital rotation)
        params[l, g, 1] = theta (pair exchange)
    n_qubits : int
        Total qubits (must be divisible by 4 for interleaved ordering).
        n_qubits = 2 * M where M = number of spatial orbitals.
    n_layers : int
        Number of fabric layers.
    pi_gate : bool
        Use Pi = QNPOR(pi) initialization (recommended).
    """
    M = n_qubits // 2  # number of spatial orbitals

    for layer in range(n_layers):
        # Even layer: gates on orbital pairs (0,1), (2,3), ...
        # Odd layer: gates on orbital pairs (1,2), (3,4), ...
        offset = layer % 2

        gate_idx = 0
        for p in range(offset, M - 1, 2):
            q = p + 1
            wires = [2 * p, 2 * p + 1, 2 * q, 2 * q + 1]

            if gate_idx < params.shape[1]:
                phi = params[layer, gate_idx, 0]
                theta = params[layer, gate_idx, 1]
                qnp_gate(phi, theta, wires, pi_gate=pi_gate)
            gate_idx += 1


def get_qnp_param_shape(n_qubits, n_layers):
    """
    Get the parameter shape for the QNP fabric.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    n_layers : int
        Number of fabric layers.

    Returns
    -------
    tuple
        (n_layers, max_gates_per_layer, 2)
    """
    M = n_qubits // 2
    max_gates = max(M // 2, (M - 1) // 2)
    return (n_layers, max_gates, 2)


def count_qnp_params(n_qubits, n_layers):
    """
    Count total trainable parameters in the QNP fabric.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    n_layers : int
        Number of fabric layers.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    shape = get_qnp_param_shape(n_qubits, n_layers)
    return shape[0] * shape[1] * shape[2]


# ============================================================
# QNP VQE Ansatz
# ============================================================

def qnp_ansatz(params, wires, n_layers, hf_state=None):
    """
    Full QNP VQE ansatz: initial state preparation + QNP fabric.

    Parameters
    ----------
    params : array
        Variational parameters for the QNP fabric.
    wires : list of int
        Qubit wires.
    n_layers : int
        Number of fabric layers.
    hf_state : list of int, optional
        Initial reference state (e.g., [1,1,1,1,0,0,0,0] for 4 electrons
        in 8 qubits).
    """
    n_qubits = len(wires)

    if hf_state is not None:
        qml.BasisState(np.array(hf_state), wires=wires)

    param_shape = get_qnp_param_shape(n_qubits, n_layers)
    params_reshaped = params.reshape(param_shape)
    qnp_fabric(params_reshaped, n_qubits, n_layers, pi_gate=True)


def create_qnp_vqe_circuit(hamiltonian, n_qubits, n_layers, hf_state):
    """
    Create a VQE QNode using the QNP ansatz.

    Parameters
    ----------
    hamiltonian : qml.Hamiltonian
        Hamiltonian operator.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of QNP layers.
    hf_state : list of int
        Reference state.

    Returns
    -------
    cost_fn : QNode
        VQE cost function.
    n_params : int
        Number of variational parameters.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def cost_fn(params):
        qnp_ansatz(params, wires=range(n_qubits), n_layers=n_layers, hf_state=hf_state)
        return qml.expval(hamiltonian)

    param_shape = get_qnp_param_shape(n_qubits, n_layers)
    n_params = param_shape[0] * param_shape[1] * param_shape[2]

    return cost_fn, n_params


# ============================================================
# Parameter Initialization
# ============================================================

def initialize_qnp_params(n_qubits, n_layers, strategy="A"):
    """
    Initialize QNP parameters following recommended strategies.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits.
    n_layers : int
        Number of fabric layers.
    strategy : str
        'A' (recommended): phi = pi/2, theta = 0, with Pi = QNPOR(pi).
            Populates all light cones from the start, avoids barren plateaus.
        'B': phi = pi, theta = 0, with Pi = I.
            Shallower circuits but may encounter plateaus.
        'random': Small random initialization.

    Returns
    -------
    pnp.ndarray
        Initialized parameters with requires_grad=True.
    """
    shape = get_qnp_param_shape(n_qubits, n_layers)

    if strategy == "A":
        params = np.zeros(shape)
        params[:, :, 0] = np.pi / 2  # phi = pi/2 for orbital rotations
        params[:, :, 1] = 0.0  # theta = 0 for pair exchange
    elif strategy == "B":
        params = np.zeros(shape)
        params[:, :, 0] = np.pi
        params[:, :, 1] = 0.0
    else:  # random
        params = np.random.uniform(-0.1, 0.1, shape)

    return pnp.array(params, requires_grad=True)


# ============================================================
# Hardware-Efficient Ansatz (for comparison)
# ============================================================

def hardware_efficient_ansatz(params, wires, n_layers):
    """
    Standard hardware-efficient ansatz for comparison.
    Uses StronglyEntanglingLayers -- does NOT preserve quantum numbers.

    Parameters
    ----------
    params : array
        Variational parameters.
    wires : list of int
        Qubit wires.
    n_layers : int
        Number of entangling layers (unused here as params encode shape).
    """
    qml.StronglyEntanglingLayers(params, wires=wires)


def create_hea_vqe_circuit(hamiltonian, n_qubits, n_layers, hf_state):
    """
    Create a VQE QNode with hardware-efficient ansatz.

    Parameters
    ----------
    hamiltonian : qml.Hamiltonian
        Hamiltonian operator.
    n_qubits : int
        Number of qubits.
    n_layers : int
        Number of entangling layers.
    hf_state : list of int
        Reference state.

    Returns
    -------
    cost_fn : QNode
        VQE cost function.
    n_params : int
        Number of variational parameters.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd")
    def cost_fn(params):
        qml.BasisState(np.array(hf_state), wires=range(n_qubits))
        qml.StronglyEntanglingLayers(
            params.reshape(n_layers, n_qubits, 3), wires=range(n_qubits)
        )
        return qml.expval(hamiltonian)

    n_params = n_layers * n_qubits * 3
    return cost_fn, n_params
