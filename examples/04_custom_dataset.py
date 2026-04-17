"""
Custom dataset: Bring your own data and compare QML vs ML.
"""
import numpy as np
from qmc import Benchmark

# Generate a synthetic dataset
np.random.seed(42)
n_samples = 500
X = np.random.randn(n_samples, 6)
# Create a nonlinear decision boundary
y = ((X[:, 0] * X[:, 1] + X[:, 2] ** 2) > 0.5).astype(int)

print(f"Dataset: {n_samples} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"Class balance: {np.mean(y==0):.0%} / {np.mean(y==1):.0%}")

# Run comparison
bench = Benchmark(
    dataset=(X, y),
    classical_methods=["RF", "SVM", "MLP"],
    n_qubits=6,
    n_layers=3,
)

results = bench.run()
print("\n" + bench.summary())
bench.report("custom_results/")
