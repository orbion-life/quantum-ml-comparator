"""
Molecular Hamiltonians for Quantum Chemistry Benchmarks
=======================================================
Standard benchmark molecules for VQE and quantum chemistry comparisons.
Uses PennyLane's qchem module to build molecular Hamiltonians.

Benchmark molecules:
  - H2 (simplest test, 4 qubits in STO-3G)
  - HeH+ (2 electrons, 4 qubits)
  - LiH (metal hydride, 6 qubits with active space)
  - H2O (water, 8 qubits with active space)
  - H2_stretched (strongly correlated regime, 4 qubits)
"""

import numpy as np


# Molecular geometries in Bohr (atomic units)

MOLECULES = {
    'H2': {
        'symbols': ['H', 'H'],
        'coordinates': np.array([0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.4]),  # 1.4 Bohr = 0.74 Angstrom
        'charge': 0,
        'multiplicity': 1,
        'description': 'Hydrogen molecule - simplest test case',
        'n_electrons': 2,
        'active_electrons': 2,
        'active_orbitals': 2,  # -> 4 qubits
    },
    'HeH+': {
        'symbols': ['He', 'H'],
        'coordinates': np.array([0.0, 0.0, 0.0,
                                  0.0, 0.0, 1.4633]),
        'charge': 1,
        'multiplicity': 1,
        'description': 'Helium hydride cation - 2-electron system',
        'n_electrons': 2,
        'active_electrons': 2,
        'active_orbitals': 2,  # -> 4 qubits
    },
    'LiH': {
        'symbols': ['Li', 'H'],
        'coordinates': np.array([0.0, 0.0, 0.0,
                                  0.0, 0.0, 3.0155]),  # 1.596 Angstrom
        'charge': 0,
        'multiplicity': 1,
        'description': 'Lithium hydride - standard quantum chemistry benchmark',
        'n_electrons': 4,
        'active_electrons': 2,
        'active_orbitals': 3,  # -> 6 qubits (freeze core)
    },
    'H2O': {
        'symbols': ['O', 'H', 'H'],
        'coordinates': np.array([0.0, 0.0, 0.2217,
                                  0.0, 1.4309, -0.8867,
                                  0.0, -1.4309, -0.8867]),
        'charge': 0,
        'multiplicity': 1,
        'description': 'Water molecule - multi-electron benchmark',
        'n_electrons': 10,
        'active_electrons': 4,
        'active_orbitals': 4,  # -> 8 qubits (freeze core)
    },
    'H2_stretched': {
        'symbols': ['H', 'H'],
        'coordinates': np.array([0.0, 0.0, 0.0,
                                  0.0, 0.0, 4.0]),  # Stretched bond
        'charge': 0,
        'multiplicity': 1,
        'description': 'Stretched H2 - strong correlation test',
        'n_electrons': 2,
        'active_electrons': 2,
        'active_orbitals': 2,  # -> 4 qubits
    },
}


def build_hamiltonian(mol_name, basis='sto-3g'):
    """
    Build molecular Hamiltonian using PennyLane qchem.

    Parameters
    ----------
    mol_name : str
        Key from MOLECULES dict.
    basis : str
        Basis set (default: sto-3g for small qubit counts).

    Returns
    -------
    dict with:
        'hamiltonian': qml.Hamiltonian
        'n_qubits': int
        'n_electrons': int
        'hf_state': list of int
        'hf_energy': float
        'fci_energy': float (if computable)
        'molecule_info': dict
    """
    import pennylane as qml

    mol = MOLECULES[mol_name]
    symbols = mol['symbols']
    coords = mol['coordinates']
    charge = mol['charge']
    mult = mol['multiplicity']

    print(f"  Building Hamiltonian for {mol_name}: {mol['description']}")

    # Build the Hamiltonian
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        symbols, coords,
        charge=charge,
        mult=mult,
        basis=basis,
        mapping='jordan_wigner',
    )

    n_electrons = mol['n_electrons']

    # Active space reduction if specified
    active_e = mol.get('active_electrons', n_electrons)
    active_o = mol.get('active_orbitals', n_qubits // 2)

    if active_e < n_electrons or active_o < n_qubits // 2:
        print(f"    Full space: {n_qubits} qubits, {n_electrons} electrons")
        try:
            H, active_qubits_out = qml.qchem.molecular_hamiltonian(
                symbols, coords,
                charge=charge,
                mult=mult,
                basis=basis,
                mapping='jordan_wigner',
                active_electrons=active_e,
                active_orbitals=active_o,
            )
            n_qubits = active_qubits_out
            n_electrons = active_e
            print(f"    Active space: ({active_e}e, {active_o}o) -> {n_qubits} qubits")
        except Exception as e:
            print(f"    Active space failed ({e}), using full space")

    # Hartree-Fock state
    hf_state = [0] * n_qubits
    n_alpha = (n_electrons + 1) // 2
    n_beta = n_electrons // 2
    for i in range(n_alpha):
        hf_state[2 * i] = 1      # alpha electrons
    for i in range(n_beta):
        hf_state[2 * i + 1] = 1  # beta electrons

    # HF energy
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def hf_circuit():
        qml.BasisState(np.array(hf_state), wires=range(n_qubits))
        return qml.expval(H)

    hf_energy = float(hf_circuit())

    # FCI energy (exact diagonalization for small systems)
    fci_energy = None
    if n_qubits <= 16:
        try:
            H_matrix = qml.matrix(H)
            eigenvalues = np.linalg.eigvalsh(H_matrix)
            fci_energy = float(eigenvalues[0])
            print(f"    HF energy:  {hf_energy:.6f} Ha")
            print(f"    FCI energy: {fci_energy:.6f} Ha")
            print(f"    Correlation energy: {fci_energy - hf_energy:.6f} Ha")
        except Exception as e:
            print(f"    FCI computation failed: {e}")

    result = {
        'hamiltonian': H,
        'n_qubits': n_qubits,
        'n_electrons': n_electrons,
        'hf_state': hf_state,
        'hf_energy': hf_energy,
        'fci_energy': fci_energy,
        'molecule_info': mol,
        'basis': basis,
    }

    print(f"    Qubits: {n_qubits}, Electrons: {n_electrons}")
    n_terms = len(H.operands) if hasattr(H, 'operands') else (
        len(H.ops) if hasattr(H, 'ops') else '?')
    print(f"    Hamiltonian terms: {n_terms}")

    return result


def get_all_molecules(max_qubits=12):
    """Build Hamiltonians for all molecules up to max_qubits."""
    results = {}
    for name in MOLECULES:
        mol = MOLECULES[name]
        expected_qubits = 2 * mol.get('active_orbitals', mol['n_electrons'])
        if expected_qubits <= max_qubits:
            try:
                results[name] = build_hamiltonian(name)
            except Exception as e:
                print(f"  FAILED {name}: {e}")
    return results


class MoleculeLibrary:
    """High-level interface to the benchmark molecule library."""

    @classmethod
    def list(cls):
        """Return list of available molecule names."""
        return list(MOLECULES.keys())

    @classmethod
    def get(cls, name):
        """
        Return the raw molecule specification dict.

        Parameters
        ----------
        name : str
            Molecule name (e.g. 'H2', 'LiH').

        Returns
        -------
        dict with symbols, coordinates, charge, multiplicity, etc.

        Raises
        ------
        KeyError
            If molecule name is not found.
        """
        if name not in MOLECULES:
            available = ', '.join(MOLECULES.keys())
            raise KeyError(
                f"Unknown molecule '{name}'. Available: {available}"
            )
        return MOLECULES[name]

    @classmethod
    def build(cls, name, basis='sto-3g'):
        """
        Build the full Hamiltonian for a molecule.

        Parameters
        ----------
        name : str
            Molecule name.
        basis : str
            Basis set (default: 'sto-3g').

        Returns
        -------
        dict with hamiltonian, n_qubits, hf_energy, fci_energy, etc.
        """
        return build_hamiltonian(name, basis=basis)
