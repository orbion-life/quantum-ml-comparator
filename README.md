# quantum-ml-comparator

[![tests](https://github.com/orbion-life/quantum-ml-comparator/actions/workflows/test.yml/badge.svg)](https://github.com/orbion-life/quantum-ml-comparator/actions/workflows/test.yml)
[![lint](https://github.com/orbion-life/quantum-ml-comparator/actions/workflows/lint.yml/badge.svg)](https://github.com/orbion-life/quantum-ml-comparator/actions/workflows/lint.yml)
[![PyPI version](https://img.shields.io/pypi/v/quantum-ml-comparator.svg)](https://pypi.org/project/quantum-ml-comparator/)
[![Python versions](https://img.shields.io/pypi/pyversions/quantum-ml-comparator.svg)](https://pypi.org/project/quantum-ml-comparator/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![codecov](https://codecov.io/gh/orbion-life/quantum-ml-comparator/branch/main/graph/badge.svg)](https://codecov.io/gh/orbion-life/quantum-ml-comparator)

> Developed at [Orbion GmbH](https://orbion.life).

**Compare quantum machine learning algorithms against classical ML — with automatic QML recommendations.**

A general-purpose, open-source framework to benchmark QML vs classical ML on your own datasets. Tell it what classical algorithm you're using, and it recommends which quantum algorithms to compare against, explains why, and runs the comparison for you.

## Install

```bash
pip install -e .
```

For molecular VQE support:
```bash
pip install -e ".[molecules]"
```

## Quickstart (3 lines)

```python
from qmc import Benchmark

bench = Benchmark(dataset="iris", classical_methods=["MLP", "SVM", "RF"])
bench.run()
bench.report("results/")
```

Quantum methods are **auto-recommended** based on your classical methods.

## The QML Recommender

Don't know which quantum algorithm to try? Ask:

```python
from qmc import print_recommendations

print_recommendations("RandomForest")
```

Output:
```
[PRIMARY] Quantum Kernel Ensemble  (difficulty: medium)
  Ensemble of quantum-kernel SVMs on bootstrap samples, mimicking Random Forest's bagging.
  Rationale: Combining multiple quantum kernel models reduces variance, similar to how
             Random Forest aggregates decision trees.
  Circuit:   8 qubits, 4 layers

[SECONDARY] VQC  (difficulty: easy)
  Variational Quantum Classifier as a single strong learner replacing the tree ensemble.
  Rationale: A sufficiently expressive VQC can match an ensemble of weak learners.
```

Supported classical algorithms: **SVM, MLP, Random Forest, Logistic Regression, k-NN, XGBoost, Naive Bayes, PCA** (plus any algorithm falls back to general-purpose VQC/QuantumKernel).

## What's Included

### Classical models
MLP (PyTorch), SVM, Random Forest, Logistic Regression, k-NN, Gradient Boosting, Naive Bayes, Decision Tree.

### Quantum circuits
- **VQC** — Variational Quantum Classifier (binary + multiclass)
- **Quantum Kernel** — IQP-style feature map + precomputed SVM
- **QNP ansatz** — Particle-number-preserving gates (Anselmetti et al.)
- **HEA ansatz** — StronglyEntanglingLayers (generic)
- Plus circuit factory templates for custom designs

### Molecular VQE
Run VQE on standard benchmark molecules (H2, HeH+, LiH, H2O) with QNP or HEA ansatze. Compare ansatz performance:

```python
from qmc.molecules import VQERunner

runner = VQERunner(molecule="H2", ansatz="QNP", n_layers=4)
result = runner.run()
print(f"Energy: {result.energy:.6f} Ha  Error: {result.error:.2e}")
```

### Live dashboard
```python
from qmc.dashboard import start_dashboard
start_dashboard(port=8501)
# Open http://localhost:8501 — live training curves during bench.run()
```

## Bring Your Own Data

```python
import numpy as np
from qmc import Benchmark

X = np.random.randn(500, 6)
y = (X[:, 0] * X[:, 1] > 0).astype(int)

bench = Benchmark(dataset=(X, y), classical_methods=["RF", "MLP"])
bench.run()
```

Or from a CSV:
```python
bench = Benchmark(dataset="data.csv", target_column="label")
```

Built-in datasets: `iris`, `breast_cancer`, `wine`, `digits`, `moons`, `circles`, `blobs`.

## Example Output

Running `examples/01_quickstart.py` on Iris:

| Rank | Method | Type | Accuracy | F1 | Time |
|------|--------|------|----------|-----|------|
| 1 | VQC | quantum | 1.0000 | 1.0000 | 340.7s |
| 2 | QuantumKernel | quantum | 0.9556 | 0.9554 | 7.8s |
| 3 | MLP | classical | 0.9333 | 0.9333 | 0.06s |
| 4 | SVM | classical | 0.9111 | 0.9107 | 0.001s |
| 5 | RF | classical | 0.8889 | 0.8878 | 0.03s |

## Examples

- `examples/01_quickstart.py` — Classical vs quantum on Iris
- `examples/02_recommender.py` — Get QML recommendations
- `examples/03_molecule_vqe.py` — VQE on H2 (requires PySCF)
- `examples/04_custom_dataset.py` — Bring your own numpy data

## Run Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

## Mappings Reference

| Your classical algorithm | Recommended quantum counterpart |
|--------------------------|--------------------------------|
| SVM | Quantum Kernel SVM, VQC |
| MLP / Neural Net | VQC, Data Re-uploading VQC |
| Random Forest | Quantum Kernel Ensemble, VQC |
| Logistic Regression | Quantum Kernel + Linear SVM |
| k-NN | Quantum Kernel k-NN |
| XGBoost | Quantum Kernel SVM, Quantum Boosted Ensemble |
| Naive Bayes | VQC with probabilistic readout |
| PCA | Quantum feature map, Quantum Autoencoder |
| *anything else* | VQC, Quantum Kernel (general-purpose) |

## License

MIT — use it anywhere.

## Acknowledgments

This repository was developed with AI coding assistance. The research direction, experimental design, verification, and technical decisions are original work; the code scaffolding was accelerated with Claude.

QNP gate implementation based on [Anselmetti et al. (2021)](https://arxiv.org/abs/2104.05695).
Built on [PennyLane](https://pennylane.ai/) and [scikit-learn](https://scikit-learn.org/).
