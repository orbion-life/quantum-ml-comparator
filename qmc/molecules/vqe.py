"""
VQE Runner for Quantum Chemistry Benchmarks
============================================
Run Variational Quantum Eigensolver (VQE) on benchmark molecules.
Compare ansatz convergence to FCI ground state energy.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np


@dataclass
class VQEResult:
    """Result from a single VQE run."""

    energy: float
    error: Optional[float]
    n_steps: int
    n_params: int
    energies: List[float] = field(default_factory=list)
    best_params: Optional[np.ndarray] = None
    wall_time: float = 0.0
    molecule: str = ""
    ansatz: str = ""
    n_layers: int = 0
    fci_energy: Optional[float] = None
    hf_energy: Optional[float] = None


def run_vqe(cost_fn, init_params, n_params, max_steps=200, lr=0.1,
            conv_threshold=1e-6, callback=None):
    """
    Run VQE optimization.

    Parameters
    ----------
    cost_fn : callable
        VQE cost function (QNode).
    init_params : array
        Initial parameters.
    n_params : int
        Number of parameters.
    max_steps : int
        Maximum optimization steps.
    lr : float
        Learning rate.
    conv_threshold : float
        Convergence threshold for energy change.
    callback : callable, optional
        Called each step with (step, energy, params).

    Returns
    -------
    dict with energies, params, convergence history.
    """
    import pennylane as qml
    from pennylane import numpy as pnp

    params = pnp.array(init_params.flatten()[:n_params], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=lr)

    energies = []
    best_energy = float('inf')
    best_params = params.copy()

    for step in range(max_steps):
        params, energy = opt.step_and_cost(cost_fn, params)
        energy = float(energy)
        energies.append(energy)

        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()

        if callback:
            callback(step, energy, params)

        # Convergence check
        if len(energies) > 5:
            recent_change = abs(energies[-1] - energies[-5])
            if recent_change < conv_threshold:
                break

    return {
        'energies': energies,
        'best_energy': best_energy,
        'best_params': best_params,
        'n_steps': len(energies),
        'n_params': n_params,
    }


def compare_ansatze(mol_name, n_layers_list=None, max_steps=150):
    """
    Compare QNP vs HEA ansatze on a single molecule.

    Parameters
    ----------
    mol_name : str
        Molecule name from MOLECULES dict.
    n_layers_list : list of int
        Layer counts to test.
    max_steps : int
        Max VQE steps per configuration.

    Returns
    -------
    dict with results per method per layer count.
    """
    from qmc.molecules.library import build_hamiltonian, MOLECULES

    if n_layers_list is None:
        n_layers_list = [2, 4, 8, 12]

    mol_data = build_hamiltonian(mol_name)
    H = mol_data['hamiltonian']
    n_qubits = mol_data['n_qubits']
    n_electrons = mol_data['n_electrons']
    hf_state = mol_data['hf_state']
    hf_energy = mol_data['hf_energy']
    fci_energy = mol_data['fci_energy']

    print(f"\n{'=' * 60}")
    print(f"  VQE Comparison: {mol_name}")
    print(f"  {n_qubits} qubits, {n_electrons} electrons")
    print(f"  HF = {hf_energy:.6f}, FCI = {fci_energy:.6f}")
    if fci_energy is not None:
        print(f"  Correlation energy = {fci_energy - hf_energy:.6f} Ha")
    print(f"{'=' * 60}")

    results = {
        'molecule': mol_name,
        'n_qubits': n_qubits,
        'n_electrons': n_electrons,
        'hf_energy': hf_energy,
        'fci_energy': fci_energy,
        'methods': {},
    }

    # Try to import QNP gates; they may not be available in all setups
    try:
        from qmc.circuits.qnp_gates import (
            create_qnp_vqe_circuit, initialize_qnp_params,
            create_hea_vqe_circuit,
        )
        has_qnp = True
    except ImportError:
        has_qnp = False
        print("  Warning: qmc.circuits.qnp_gates not found, skipping QNP/HEA")

    if not has_qnp:
        return results

    for n_layers in n_layers_list:
        print(f"\n  --- {n_layers} layers ---")

        # 1. QNP Ansatz
        print(f"  [QNP] ", end='', flush=True)
        try:
            qnp_cost, qnp_n_params = create_qnp_vqe_circuit(
                H, n_qubits, n_layers, hf_state
            )
            qnp_init = initialize_qnp_params(n_qubits, n_layers, strategy='A')

            start = time.time()
            qnp_result = run_vqe(
                qnp_cost, qnp_init, qnp_n_params,
                max_steps=max_steps, lr=0.1,
            )
            qnp_time = time.time() - start

            error = (qnp_result['best_energy'] - fci_energy
                     if fci_energy is not None else None)
            print(f"{qnp_n_params} params, E={qnp_result['best_energy']:.6f}, "
                  f"error={error:.2e} Ha, {qnp_result['n_steps']} steps, "
                  f"{qnp_time:.1f}s")

            key = f'QNP_L{n_layers}'
            results['methods'][key] = {
                'method': 'QNP',
                'n_layers': n_layers,
                'n_params': qnp_n_params,
                'best_energy': qnp_result['best_energy'],
                'error': error,
                'energies': [float(e) for e in qnp_result['energies']],
                'n_steps': qnp_result['n_steps'],
                'time': qnp_time,
            }
        except Exception as e:
            print(f"FAILED: {e}")

        # 2. Hardware-Efficient Ansatz
        print(f"  [HEA] ", end='', flush=True)
        try:
            hea_cost, hea_n_params = create_hea_vqe_circuit(
                H, n_qubits, n_layers, hf_state
            )
            hea_init = np.random.uniform(-0.1, 0.1, hea_n_params)

            start = time.time()
            hea_result = run_vqe(
                hea_cost, hea_init, hea_n_params,
                max_steps=max_steps, lr=0.1,
            )
            hea_time = time.time() - start

            error = (hea_result['best_energy'] - fci_energy
                     if fci_energy is not None else None)
            print(f"{hea_n_params} params, E={hea_result['best_energy']:.6f}, "
                  f"error={error:.2e} Ha, {hea_result['n_steps']} steps, "
                  f"{hea_time:.1f}s")

            key = f'HEA_L{n_layers}'
            results['methods'][key] = {
                'method': 'HEA',
                'n_layers': n_layers,
                'n_params': hea_n_params,
                'best_energy': hea_result['best_energy'],
                'error': error,
                'energies': [float(e) for e in hea_result['energies']],
                'n_steps': hea_result['n_steps'],
                'time': hea_time,
            }
        except Exception as e:
            print(f"FAILED: {e}")

    return results


class VQERunner:
    """
    High-level VQE runner for benchmark molecules.

    Usage
    -----
    >>> runner = VQERunner(molecule="H2", ansatz="QNP", n_layers=4)
    >>> result = runner.run()
    >>> print(result.energy, result.error, result.n_steps)
    """

    SUPPORTED_ANSATZE = ('QNP', 'HEA')

    def __init__(self, molecule="H2", ansatz="QNP", n_layers=4,
                 max_steps=200, lr=0.1, conv_threshold=1e-6, basis='sto-3g'):
        """
        Parameters
        ----------
        molecule : str
            Molecule name (e.g. 'H2', 'LiH', 'HeH+').
        ansatz : str
            Ansatz type: 'QNP' or 'HEA'.
        n_layers : int
            Number of ansatz layers.
        max_steps : int
            Maximum VQE optimization steps.
        lr : float
            Learning rate for gradient descent.
        conv_threshold : float
            Convergence threshold.
        basis : str
            Basis set for Hamiltonian construction.
        """
        if ansatz not in self.SUPPORTED_ANSATZE:
            raise ValueError(
                f"Unsupported ansatz '{ansatz}'. "
                f"Choose from: {self.SUPPORTED_ANSATZE}"
            )

        self.molecule = molecule
        self.ansatz = ansatz
        self.n_layers = n_layers
        self.max_steps = max_steps
        self.lr = lr
        self.conv_threshold = conv_threshold
        self.basis = basis

    def run(self, callback=None):
        """
        Run VQE and return a VQEResult.

        Parameters
        ----------
        callback : callable, optional
            Called each step with (step, energy, params).

        Returns
        -------
        VQEResult with energy, error, n_steps, etc.
        """
        from qmc.molecules.library import build_hamiltonian

        mol_data = build_hamiltonian(self.molecule, basis=self.basis)
        H = mol_data['hamiltonian']
        n_qubits = mol_data['n_qubits']
        hf_state = mol_data['hf_state']
        hf_energy = mol_data['hf_energy']
        fci_energy = mol_data['fci_energy']

        # Build cost function based on ansatz
        try:
            from qmc.circuits.qnp_gates import (
                create_qnp_vqe_circuit, initialize_qnp_params,
                create_hea_vqe_circuit,
            )
        except ImportError:
            raise ImportError(
                "qmc.circuits.qnp_gates is required for VQE. "
                "Ensure the circuits module is installed."
            )

        if self.ansatz == 'QNP':
            cost_fn, n_params = create_qnp_vqe_circuit(
                H, n_qubits, self.n_layers, hf_state
            )
            init_params = initialize_qnp_params(
                n_qubits, self.n_layers, strategy='A'
            )
        elif self.ansatz == 'HEA':
            cost_fn, n_params = create_hea_vqe_circuit(
                H, n_qubits, self.n_layers, hf_state
            )
            init_params = np.random.uniform(-0.1, 0.1, n_params)
        else:
            raise ValueError(f"Unknown ansatz: {self.ansatz}")

        start = time.time()
        raw = run_vqe(
            cost_fn, init_params, n_params,
            max_steps=self.max_steps, lr=self.lr,
            conv_threshold=self.conv_threshold,
            callback=callback,
        )
        wall_time = time.time() - start

        error = (raw['best_energy'] - fci_energy
                 if fci_energy is not None else None)

        return VQEResult(
            energy=raw['best_energy'],
            error=error,
            n_steps=raw['n_steps'],
            n_params=raw['n_params'],
            energies=raw['energies'],
            best_params=raw['best_params'],
            wall_time=wall_time,
            molecule=self.molecule,
            ansatz=self.ansatz,
            n_layers=self.n_layers,
            fci_energy=fci_energy,
            hf_energy=hf_energy,
        )
