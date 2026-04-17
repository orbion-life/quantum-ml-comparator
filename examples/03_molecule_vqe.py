"""
VQE on molecules: Compare QNP vs HEA ansatze on H2.

Requires: pip install quantum-ml-comparator[molecules]
"""
try:
    from qmc.molecules import MoleculeLibrary, VQERunner

    # List available molecules
    print("Available molecules:", MoleculeLibrary.list())

    # Run VQE on H2 with QNP ansatz
    runner = VQERunner(molecule="H2", ansatz="QNP", n_layers=4)
    result = runner.run(max_steps=100)

    print(f"\nH2 VQE Result (QNP, 4 layers):")
    print(f"  Energy: {result.energy:.6f} Ha")
    print(f"  Error from FCI: {result.error:.6f} Ha")
    print(f"  Steps: {result.n_steps}")
    print(f"  Parameters: {result.n_params}")

    # Compare QNP vs HEA
    print("\n--- Comparing QNP vs HEA ---")
    for ansatz in ["QNP", "HEA"]:
        runner = VQERunner(molecule="H2", ansatz=ansatz, n_layers=4)
        result = runner.run(max_steps=100)
        print(f"  {ansatz}: E={result.energy:.6f}, err={result.error:.2e}, "
              f"params={result.n_params}, steps={result.n_steps}")

except ImportError:
    print("PySCF not installed. Run: pip install quantum-ml-comparator[molecules]")
