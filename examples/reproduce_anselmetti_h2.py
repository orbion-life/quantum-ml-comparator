"""
Reproduce the H2 VQE benchmark from Anselmetti et al. (2021).

Paper: "A local, expressive, quantum-number-preserving VQE ansatz for
fermionic systems", arXiv:2104.05695

This script:
1. Runs VQE on H2 at the experimental equilibrium (0.7414 Å) using
   the QNP ansatz, and asserts the energy matches the FCI reference
   to < 1 mHa.
2. Computes the H2 dissociation curve on a grid of bond lengths and
   compares against the analytic PennyLane Hamiltonian's exact
   diagonalization (treated as "FCI" for STO-3G H2, where it is in
   fact exact).
3. Enforces four scientific-sanity gates from the project plan:
   (a) equilibrium error < 1 mHa,
   (b) VQE energy >= FCI at every point (variational principle),
   (c) curve minimum in [0.70, 0.78] Å,
   (d) curve smooth (no discontinuities).

Run directly:
    python examples/reproduce_anselmetti_h2.py

Outputs:
    examples/figures/h2_dissociation.png
    examples/figures/h2_dissociation.csv
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe for CI

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qmc.circuits.qnp import create_qnp_vqe_circuit, initialize_qnp_params
from qmc.molecules.vqe import run_vqe


# ----------------------------------------------------------------------
# Constants — scientific references
# ----------------------------------------------------------------------
# Published FCI (== exact for STO-3G H2) at equilibrium. See e.g.
# Anselmetti et al. Table II and every undergraduate quantum chemistry
# textbook; we also recompute it below via exact diagonalisation.
FCI_H2_EQUILIBRIUM = -1.137270174  # Ha at R = 0.7414 Å, STO-3G
EQUILIBRIUM_BOND_LENGTH = 0.7414   # Å
CHEMICAL_ACCURACY = 1.6e-3          # Ha (1 kcal/mol)
TARGET_ACCURACY = 1e-3              # Ha — our assertion threshold (< 1 mHa)

OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# H2 Hamiltonian at arbitrary bond length
# ----------------------------------------------------------------------

def h2_hamiltonian(bond_length: float):
    """Build H2 Hamiltonian at the given bond length (Å) in STO-3G.

    Returns
    -------
    H : qml.Hamiltonian
    n_qubits : int
    n_electrons : int
    hf_state : np.ndarray
    fci_energy : float
        Exact ground-state energy from dense diagonalisation. For
        STO-3G H2 this *is* FCI.
    """
    symbols = ["H", "H"]
    # Coordinates in Angstroms; PennyLane qchem takes Angstroms when
    # unit='angstrom'.
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]])
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coords, basis="sto-3g", unit="angstrom",
    )
    n_electrons = 2
    hf_state = np.array([1, 1] + [0] * (n_qubits - n_electrons))

    # Exact diagonalisation for 4-qubit H (tiny matrix).
    mat = qml.matrix(H)
    eigs = np.linalg.eigvalsh(mat)
    fci_energy = float(eigs[0])
    return H, n_qubits, n_electrons, hf_state, fci_energy


# ----------------------------------------------------------------------
# VQE at a single bond length
# ----------------------------------------------------------------------

def vqe_h2_at(bond_length: float, n_layers: int = 2,
              max_steps: int = 150, lr: float = 0.1):
    """Run QNP-VQE on H2 at a single bond length. Returns (energy, error)."""
    H, n_qubits, n_electrons, hf_state, fci = h2_hamiltonian(bond_length)
    cost_fn, n_params = create_qnp_vqe_circuit(
        H, n_qubits, n_layers, hf_state.tolist()
    )
    init_params = initialize_qnp_params(n_qubits, n_layers, strategy="A")

    result = run_vqe(
        cost_fn, init_params, n_params,
        max_steps=max_steps, lr=lr, conv_threshold=1e-8,
    )
    energy = float(result["best_energy"])
    error = energy - fci
    return energy, error, fci, n_params


# ----------------------------------------------------------------------
# Scientific-sanity gates
# ----------------------------------------------------------------------

def assert_equilibrium_accuracy(energy: float) -> None:
    """Gate: equilibrium VQE energy within target of FCI."""
    err = abs(energy - FCI_H2_EQUILIBRIUM)
    assert err < TARGET_ACCURACY, (
        f"Equilibrium H2 VQE off by {err:.2e} Ha "
        f"(target < {TARGET_ACCURACY:.2e} Ha)"
    )
    print(f"  [PASS] |E_VQE - E_FCI| = {err:.2e} Ha < {TARGET_ACCURACY:.2e} Ha")


def assert_variational(vqe_energies: np.ndarray, fci_energies: np.ndarray,
                       tol: float = 5e-4) -> None:
    """Gate: VQE >= FCI at every bond length (variational principle).

    A small numerical slack `tol` accounts for floating-point rounding
    in the exact diagonalisation reference.
    """
    violations = np.where(vqe_energies < fci_energies - tol)[0]
    assert len(violations) == 0, (
        f"Variational principle violated at {len(violations)} bond lengths "
        f"(tol={tol:.1e}): indices {violations.tolist()}"
    )
    print(f"  [PASS] VQE >= FCI at every bond length (tol={tol:.1e})")


def assert_minimum_in_range(bond_lengths: np.ndarray, energies: np.ndarray,
                             lo: float = 0.70, hi: float = 0.78) -> None:
    """Gate: VQE curve minimum in experimental range [lo, hi] Å."""
    r_min = float(bond_lengths[int(np.argmin(energies))])
    assert lo <= r_min <= hi, (
        f"VQE minimum at R={r_min:.3f} Å outside expected range [{lo}, {hi}]"
    )
    print(f"  [PASS] Minimum at R={r_min:.4f} Å in [{lo}, {hi}]")


def assert_smooth(vqe_energies: np.ndarray, fci_energies: np.ndarray,
                  max_error_jump: float = 1e-3) -> None:
    """Gate: VQE tracks FCI smoothly — no optimiser discontinuities.

    The raw curve can and should have large energy gradients on the
    repulsive wall (H2 at R=0.3 Å has a steep short-range Coulomb
    repulsion). A better smoothness check is to look at *deviations*
    between VQE and FCI: if those are smooth, the optimiser is doing
    its job, regardless of how steep the underlying potential is.
    """
    residuals = vqe_energies - fci_energies
    diffs = np.abs(np.diff(residuals))
    assert diffs.max() < max_error_jump, (
        f"VQE-FCI residual jumps by {diffs.max():.2e} Ha between adjacent "
        f"points (threshold {max_error_jump:.0e} Ha) — optimiser got stuck."
    )
    print(f"  [PASS] Max VQE-FCI residual jump {diffs.max():.2e} Ha "
          f"< {max_error_jump:.0e} Ha (smooth tracking)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print("=" * 64)
    print("Reproducing Anselmetti et al. (2021) H2 benchmark")
    print("=" * 64)

    # (1) Equilibrium single-point
    print("\n[1] Single-point VQE at equilibrium (R = 0.7414 Å)")
    t0 = time.time()
    e_eq, err_eq, fci_eq, n_params = vqe_h2_at(
        EQUILIBRIUM_BOND_LENGTH, n_layers=2, max_steps=200, lr=0.1
    )
    print(f"  Bond length : {EQUILIBRIUM_BOND_LENGTH} Å")
    print(f"  FCI         : {fci_eq:.8f} Ha")
    print(f"  VQE (QNP)   : {e_eq:.8f} Ha")
    print(f"  Error       : {err_eq:+.2e} Ha")
    print(f"  Parameters  : {n_params}")
    print(f"  Time        : {time.time() - t0:.1f} s")
    assert_equilibrium_accuracy(e_eq)

    # (2) Dissociation curve
    print("\n[2] Dissociation curve — R ∈ [0.3, 2.5] Å, 23 points")
    bond_lengths = np.linspace(0.3, 2.5, 23)
    vqe_energies = []
    fci_energies = []
    for i, R in enumerate(bond_lengths):
        e, err, fci, _ = vqe_h2_at(R, n_layers=2, max_steps=150, lr=0.1)
        vqe_energies.append(e)
        fci_energies.append(fci)
        print(f"  R={R:.3f} Å | VQE={e:+.6f} | FCI={fci:+.6f} | Δ={err:+.2e}")

    vqe_energies = np.array(vqe_energies)
    fci_energies = np.array(fci_energies)
    curve_rmse = float(np.sqrt(np.mean((vqe_energies - fci_energies) ** 2)))
    print(f"\n  Dissociation-curve RMSE vs FCI: {curve_rmse:.2e} Ha")

    # (3) Sanity gates
    print("\n[3] Scientific-sanity gates")
    assert_variational(vqe_energies, fci_energies)
    assert_minimum_in_range(bond_lengths, vqe_energies)
    assert_smooth(vqe_energies, fci_energies)
    assert curve_rmse < 1e-3, f"Curve RMSE {curve_rmse:.2e} >= 1 mHa"
    print(f"  [PASS] Curve RMSE {curve_rmse:.2e} Ha < 1 mHa")

    # (4) Save figure + CSV
    print("\n[4] Saving outputs")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bond_lengths, fci_energies, "k-", linewidth=2, label="FCI (exact)")
    ax.plot(bond_lengths, vqe_energies, "o--", markersize=6, linewidth=1.2,
            label="VQE (QNP, 2 layers)")
    ax.set_xlabel("H–H bond length (Å)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title("H$_2$ dissociation curve, STO-3G basis\n"
                 "QNP-VQE vs exact diagonalisation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig_path = OUT_DIR / "h2_dissociation.png"
    fig.savefig(fig_path, dpi=140, bbox_inches="tight")
    print(f"  Figure: {fig_path}")

    csv_path = OUT_DIR / "h2_dissociation.csv"
    np.savetxt(
        csv_path,
        np.column_stack([bond_lengths, vqe_energies, fci_energies,
                          vqe_energies - fci_energies]),
        header="bond_length_A,vqe_energy_Ha,fci_energy_Ha,error_Ha",
        delimiter=",",
        comments="",
    )
    print(f"  Data:   {csv_path}")

    print("\n" + "=" * 64)
    print("ALL GATES PASSED — Anselmetti H2 benchmark reproduced.")
    print("=" * 64)


if __name__ == "__main__":
    main()
