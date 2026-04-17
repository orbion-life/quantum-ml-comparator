# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
starting from v0.1.0. Breaking changes to the public API exported from
`qmc/__init__.py` will bump the minor version until v1.0, then the major version.

## [Unreleased]

## [0.2.0] - 2026-04-17

### Added
- `VQCClassifier` — sklearn-compatible wrapper around the VQC
  (`BaseEstimator`, `ClassifierMixin`). Supports `fit`, `predict`,
  `predict_proba`, `score`, `get_params`, `set_params`. Auto-selects
  binary vs multiclass based on `y` at fit time.
- `QuantumKernelClassifier` — sklearn-compatible wrapper around the
  IQP-style quantum kernel + precomputed SVM. Same sklearn contract,
  plus stratified subsampling to keep the O(n^2) kernel tractable.
- Both estimators live in `qmc/circuits/sklearn_api.py` and are
  re-exported from the top-level package (`from qmc import VQCClassifier`).
- PEP 561 `py.typed` marker, `TypedDict` structures in `qmc.recommender`
  (`CircuitConfig`, `Recommendation`).
- `pytest --cov=qmc` in CI with Codecov upload (advisory).
- Advisory `mypy qmc/` step in CI.

### Changed
- Documentation: added `CITATION.cff`, `CHANGELOG.md`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`.

### Fixed
- `qmc.classical.models.get_logistic_regression` no longer passes
  the removed `multi_class` kwarg (sklearn ≥1.7 compatibility).

### Deprecated
- None.

### Removed
- None (this release is additive — existing users of `VQC`,
  `VQCMulticlass`, `train_vqc`, and the functional kernel API
  continue to work).

## [0.1.0] - 2026-04-17

### Added
- `Benchmark` class with a 3-line API for comparing quantum vs classical ML
  on a user-provided or built-in dataset.
- QML algorithm recommender (`qmc.recommend`) mapping eight classical
  algorithms (SVM, MLP, Random Forest, Logistic Regression, k-NN, XGBoost,
  Naive Bayes, PCA) to quantum counterparts with rationale, difficulty
  ratings, and auto-configured circuits.
- Quantum circuits: `VQC` (binary), `VQCMulticlass`, QNP ansatz
  (Anselmetti et al. 2021), HEA via StronglyEntanglingLayers, quantum
  kernel with IQP-style feature map.
- Classical baselines: MLP (PyTorch), SVM, Random Forest, Logistic
  Regression, k-NN, Gradient Boosting, Naive Bayes, Decision Tree.
- Molecular VQE runner for H2, HeH+, LiH, H2O, H2_stretched with QNP or
  HEA ansatz comparison.
- Live training dashboard with Chart.js UI.
- Built-in datasets: iris, breast_cancer, wine, digits, moons, circles,
  blobs. Plus custom numpy / CSV input support.
- Evaluation: F1, accuracy, AUC-ROC, confusion matrices, learning
  curves, ranked comparison tables.
- 32 unit tests, CI on Python 3.10 / 3.11 / 3.12, ruff lint,
  proprietary-reference scanner.
- Branch protection on `main` requiring PR + 1 approval + passing
  status checks.

[Unreleased]: https://github.com/orbion-life/quantum-ml-comparator/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/orbion-life/quantum-ml-comparator/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/orbion-life/quantum-ml-comparator/releases/tag/v0.1.0
