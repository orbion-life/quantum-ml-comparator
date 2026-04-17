# Contributing to quantum-ml-comparator

Thanks for considering a contribution. This document explains how to set up
your development environment, the project's code standards, and how to propose
a change.

## Table of contents

1. [Development setup](#development-setup)
2. [Code standards](#code-standards)
3. [Adding a new QML algorithm](#adding-a-new-qml-algorithm)
4. [Adding a new dataset](#adding-a-new-dataset)
5. [Commit conventions](#commit-conventions)
6. [Pull request checklist](#pull-request-checklist)
7. [Release process](#release-process)

## Development setup

```bash
git clone https://github.com/orbion-life/quantum-ml-comparator.git
cd quantum-ml-comparator
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install  # once .pre-commit-config.yaml is added (see ROADMAP)
```

Run tests and lint before opening a PR:

```bash
pytest tests/ -v
ruff check qmc/ tests/
```

For notebooks that touch quantum chemistry (H2, LiH, H2O):

```bash
pip install -e ".[molecules]"   # adds pyscf
```

## Code standards

- **Line length:** 100 characters (ruff-enforced).
- **Docstrings:** NumPy style. Every public function, class, and module
  gets a docstring.
- **Type hints:** required on all new public signatures. Use
  `numpy.typing.NDArray`, `torch.Tensor`, and `Literal[...]` where
  appropriate.
- **Public API:** anything exported from `qmc/__init__.py` or a submodule
  `__init__.py`. Breaking changes here bump the minor version (see
  `CHANGELOG.md`).
- **Testing:** new features need tests. Target coverage for the `qmc/`
  package is 80%+.

## Adding a new QML algorithm

1. Create `qmc/circuits/<algorithm_name>.py`.
2. Implement a class inheriting from `sklearn.base.BaseEstimator` and
   `sklearn.base.ClassifierMixin` (or `RegressorMixin`). It should expose
   `fit(X, y)`, `predict(X)`, and `predict_proba(X)`.
3. Register the class in `qmc/circuits/__init__.py`.
4. Add an entry to the `MAPPINGS` dict in `qmc/recommender.py` so the
   recommender knows when to suggest it. Include rationale, difficulty,
   and circuit config.
5. Add a test in `tests/test_circuits.py` verifying:
   - The estimator fits on 100 samples of `make_moons` within 30 s.
   - Test accuracy > 0.7 on the moons dataset.
   - `get_params()` / `set_params()` round-trip correctly.
   - `sklearn.utils.estimator_checks.check_estimator` passes (or document
     explicit xfails).
6. Document the algorithm on the MkDocs site under
   `docs/algorithms/<algorithm_name>.md` with the four standard sections:
   mathematical formulation, circuit diagram, complexity, limitations.

## Adding a new dataset

1. Add a loader function to `qmc/datasets/builtin.py`:
   ```python
   def load_<name>() -> tuple[np.ndarray, np.ndarray, DatasetMeta]:
       """<one-line description>."""
       ...
   ```
2. Register it in the `DATASETS` dict in the same module.
3. Add a row to the README's "Built-in datasets" table.
4. Add a smoke test in `tests/test_benchmark.py` confirming the loader
   returns arrays of the documented shape.
5. Datasets must be loadable without network access (bundled or generated
   from sklearn).

## Commit conventions

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` — new feature
- `fix: ...` — bug fix
- `docs: ...` — documentation only
- `test: ...` — adding or improving tests
- `refactor: ...` — internal restructuring without behavior change
- `chore: ...` — tooling, CI, dependencies
- `bench: ...` — benchmark additions or changes

Breaking changes get a `!` suffix on the type (`feat!:`) and a
`BREAKING CHANGE:` footer.

## Pull request checklist

Before requesting review:

- [ ] Tests added or updated for your change.
- [ ] `pytest tests/` passes locally.
- [ ] `ruff check qmc/ tests/` is clean.
- [ ] No proprietary data or references introduced — the CI proprietary-
      reference grep must stay green.
- [ ] Public API changes documented in `CHANGELOG.md` under `[Unreleased]`.
- [ ] If the change adds a new algorithm, docs page and test coverage
      follow the standards above.

## Release process

1. Open a `release: prepare vX.Y.Z` PR that:
   - Moves `[Unreleased]` entries in `CHANGELOG.md` into a new
     `[X.Y.Z] - YYYY-MM-DD` section.
   - Bumps `version` in `pyproject.toml`, `qmc/__init__.py`, and
     `CITATION.cff`.
2. After merge, tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
3. The tag push triggers `.github/workflows/release.yml`, which builds
   the sdist / wheel and publishes to PyPI via Trusted Publishing.

## Questions

Open a [GitHub Discussion](https://github.com/orbion-life/quantum-ml-comparator/discussions)
or email `aniruddh.goteti@orbion.life`.
