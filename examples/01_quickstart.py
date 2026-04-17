"""
Quickstart: Compare quantum vs classical ML on the Iris dataset.

This is the simplest possible usage — 3 lines to get a comparison.
"""
from qmc import Benchmark

# One-liner: compare VQC + Quantum Kernel vs MLP + SVM + RF on Iris
bench = Benchmark(
    dataset="iris",
    classical_methods=["MLP", "SVM", "RF"],
    # quantum_methods auto-recommended based on classical methods!
    n_qubits=4,  # Iris has 4 features
)

# Run the comparison
results = bench.run()

# Print one-line summary
print(bench.summary())

# Generate a full report
bench.report("quickstart_results/")
print("\nReport saved to quickstart_results/")
