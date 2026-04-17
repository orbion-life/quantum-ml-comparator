"""Quantum circuit components for QML experiments."""

from qmc.circuits.templates import (
    create_device,
    angle_encoding_circuit,
    multi_output_circuit,
    kernel_circuit,
    get_weight_shapes,
)
from qmc.circuits.vqc import VQC, VQCMulticlass, train_vqc
from qmc.circuits.kernels import (
    compute_quantum_kernel,
    compute_quantum_kernel_cross,
    train_quantum_kernel_svm,
)
from qmc.circuits.qnp import (
    qnp_orbital_rotation,
    qnp_pair_exchange,
    qnp_gate,
    qnp_fabric,
    qnp_ansatz,
    create_qnp_vqe_circuit,
    hardware_efficient_ansatz,
    create_hea_vqe_circuit,
    initialize_qnp_params,
    get_qnp_param_shape,
    count_qnp_params,
)

__all__ = [
    # Templates
    "create_device",
    "angle_encoding_circuit",
    "multi_output_circuit",
    "kernel_circuit",
    "get_weight_shapes",
    # VQC
    "VQC",
    "VQCMulticlass",
    "train_vqc",
    # Kernels
    "compute_quantum_kernel",
    "compute_quantum_kernel_cross",
    "train_quantum_kernel_svm",
    # QNP
    "qnp_orbital_rotation",
    "qnp_pair_exchange",
    "qnp_gate",
    "qnp_fabric",
    "qnp_ansatz",
    "create_qnp_vqe_circuit",
    "hardware_efficient_ansatz",
    "create_hea_vqe_circuit",
    "initialize_qnp_params",
    "get_qnp_param_shape",
    "count_qnp_params",
]
