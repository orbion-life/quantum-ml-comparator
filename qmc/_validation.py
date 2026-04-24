"""Pydantic-backed input validation for the public ``qmc`` API.

Every public entry point in :mod:`qmc` (the top-level ``recommend``,
``Benchmark``, ``FeatureChannelBenchmark``, ``VQCClassifier``,
``QuantumKernelClassifier``) validates its caller-supplied parameters
through one of the schemas defined here before doing any real work.

This addresses the ISO 27001 *sanitized-inputs-code-scanning* control:
the trust boundary of the package is the user's call to a public
function, and pydantic gives us a single, declarative place where the
shape and bounds of those calls are enforced.

The module is private (leading underscore): callers should use the
public ``qmc.*`` functions/classes, which call into here transparently.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ArrayPair = Tuple[np.ndarray, np.ndarray]
DatasetSpec = Union[str, ArrayPair]


# ---------------------------------------------------------------------------
# Bounds chosen conservatively so they accept every existing valid usage
# (see ``tests/`` and ``examples/``) while catching obvious nonsense
# (negative qubits, zero epochs, lr=NaN, ...).
# ---------------------------------------------------------------------------

MAX_QUBITS = 64        # 2**64 amplitudes is well beyond simulator memory
MAX_LAYERS = 100       # circuits past this depth aren't trainable in practice
MAX_EPOCHS = 100_000
MAX_BATCH_SIZE = 1_000_000
MAX_FEATURES = 10_000
MAX_CLASSES = 1_000
MAX_NAME_LEN = 256


def _check_finite(name: str, v: float) -> float:
    if not np.isfinite(v):
        raise ValueError(f"{name} must be a finite number, got {v}")
    return v


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------


class RecommendInput(BaseModel):
    """Validated arguments for :func:`qmc.recommender.recommend`."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    classical_algorithm: str = Field(min_length=1, max_length=MAX_NAME_LEN)
    n_features: int = Field(default=8, ge=1, le=MAX_FEATURES)
    n_classes: int = Field(default=2, ge=2, le=MAX_CLASSES)


# ---------------------------------------------------------------------------
# Sklearn-API classifiers (VQC, QuantumKernel)
# ---------------------------------------------------------------------------


class VQCHyperparameters(BaseModel):
    """Validated hyperparameters for :class:`qmc.VQCClassifier`.

    sklearn convention is that ``__init__`` only stores arguments
    verbatim; validation happens in :meth:`fit`. We therefore call
    ``VQCHyperparameters(...)`` from inside ``fit`` rather than from
    ``__init__``.
    """

    model_config = ConfigDict(extra="forbid")

    n_qubits: int = Field(ge=1, le=MAX_QUBITS)
    n_layers: int = Field(ge=1, le=MAX_LAYERS)
    epochs: int = Field(ge=1, le=MAX_EPOCHS)
    lr: float = Field(gt=0.0, le=10.0)
    batch_size: int = Field(ge=1, le=MAX_BATCH_SIZE)
    seed: int = Field(ge=0)
    device_name: str = Field(min_length=1, max_length=MAX_NAME_LEN)
    diff_method: str = Field(min_length=1, max_length=MAX_NAME_LEN)

    @field_validator("lr")
    @classmethod
    def _lr_finite(cls, v: float) -> float:
        return _check_finite("lr", v)


class QuantumKernelHyperparameters(BaseModel):
    """Validated hyperparameters for :class:`qmc.QuantumKernelClassifier`."""

    model_config = ConfigDict(extra="forbid")

    n_qubits: int = Field(ge=1, le=MAX_QUBITS)
    n_layers: int = Field(ge=1, le=MAX_LAYERS)
    seed: int = Field(ge=0)
    device_name: str = Field(min_length=1, max_length=MAX_NAME_LEN)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


class BenchmarkConfig(BaseModel):
    """Validated configuration for :class:`qmc.Benchmark`.

    Numpy-array dataset specs and non-string method lists are validated
    by separate helpers below since pydantic's ``arbitrary_types_allowed``
    interacts awkwardly with sklearn-style duck typing.
    """

    model_config = ConfigDict(extra="forbid")

    target_column: Optional[str] = Field(default=None, max_length=MAX_NAME_LEN)
    n_qubits: int = Field(default=8, ge=1, le=MAX_QUBITS)
    n_layers: int = Field(default=4, ge=1, le=MAX_LAYERS)
    test_size: float = Field(default=0.3, gt=0.0, lt=1.0)
    random_state: int = Field(default=42, ge=0)

    @field_validator("test_size")
    @classmethod
    def _test_size_finite(cls, v: float) -> float:
        return _check_finite("test_size", v)


def validate_method_list(name: str, methods: Optional[Sequence[Any]]) -> Optional[list[str]]:
    """Validate an optional list of method-name strings.

    Returns a fresh ``list[str]`` (so callers don't share state with the
    user's input) or ``None`` if input was ``None``. Raises ``ValueError``
    on duplicates, empty lists, non-string entries, or strings that don't
    match the same length/character bounds we enforce on
    :attr:`RecommendInput.classical_algorithm`.
    """
    if methods is None:
        return None
    methods_list = list(methods)
    if not methods_list:
        raise ValueError(f"{name} must be a non-empty list, got []")
    cleaned: list[str] = []
    for i, m in enumerate(methods_list):
        if not isinstance(m, str):
            raise ValueError(
                f"{name}[{i}] must be a string, got {type(m).__name__}"
            )
        m_stripped = m.strip()
        if not (1 <= len(m_stripped) <= MAX_NAME_LEN):
            raise ValueError(
                f"{name}[{i}] must be 1..{MAX_NAME_LEN} chars after strip, got {len(m_stripped)}"
            )
        cleaned.append(m_stripped)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError(f"{name} contains duplicate entries: {cleaned}")
    return cleaned


def validate_dataset_spec(spec: Any) -> DatasetSpec:
    """Validate the polymorphic ``dataset`` argument to :class:`Benchmark`.

    Accepts:

    * a non-empty string (built-in name OR a path to an existing CSV).
      The path-vs-name distinction is left to the caller (mirroring the
      existing behaviour in :meth:`Benchmark._load_dataset`).
    * a 2-tuple ``(X, y)`` of array-likes, both finite numpy arrays.

    Raises ``ValueError`` / ``TypeError`` for anything else. CSV-path
    safety is checked here too: if the string looks like a path
    (contains a ``/`` or ``\\``) we require the file to exist before
    accepting it. This prevents the case where a typo in a built-in
    dataset name gets silently treated as a missing CSV.
    """
    if isinstance(spec, str):
        s = spec.strip()
        if not s or len(s) > 4096:
            raise ValueError(f"dataset string must be 1..4096 chars, got {len(s)}")
        looks_like_path = ("/" in s) or ("\\" in s) or s.endswith(".csv")
        if looks_like_path and not os.path.isfile(s):
            raise ValueError(f"dataset looks like a path but no such file: {s}")
        return s
    if isinstance(spec, tuple) and len(spec) == 2:
        X_raw, y_raw = spec
        try:
            X = np.asarray(X_raw, dtype=np.float64)
            y = np.asarray(y_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"dataset (X, y) tuple is not array-like: {exc}") from exc
        if X.ndim != 2:
            raise ValueError(f"dataset X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"dataset y must be 1-D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(
                f"dataset X has {len(X)} rows but y has {len(y)} — must match"
            )
        if len(X) == 0:
            raise ValueError("dataset must have at least one row")
        if not np.all(np.isfinite(X)):
            raise ValueError("dataset X contains non-finite values (NaN/Inf)")
        return (X, y)
    raise TypeError(
        "dataset must be a string name, CSV path, or (X, y) tuple — "
        f"got {type(spec).__name__}"
    )


# ---------------------------------------------------------------------------
# FeatureChannelBenchmark
# ---------------------------------------------------------------------------


class FeatureChannelConfig(BaseModel):
    """Validated scalar config for :class:`qmc.FeatureChannelBenchmark`.

    The ``channels`` mapping and the y-label arrays are validated in the
    class itself (numpy/sequence shape checks), since pydantic on its own
    is not the right tool for verifying matched array dimensions.
    """

    model_config = ConfigDict(extra="forbid")

    stratified: bool = True
    seed: int = Field(default=42, ge=0)

    @field_validator("seed")
    @classmethod
    def _seed_in_int32(cls, v: int) -> int:
        # Common upstream RNG seed limit; clamp early with a clear message.
        if v > 2**32 - 1:
            raise ValueError(f"seed must fit in uint32, got {v}")
        return v


def validate_training_sizes(sizes: Optional[Sequence[Any]], n_train: int) -> Optional[list[int]]:
    """Validate the optional ``training_sizes`` list."""
    if sizes is None:
        return None
    sizes_list = list(sizes)
    if not sizes_list:
        raise ValueError("training_sizes must be a non-empty list, got []")
    cleaned: list[int] = []
    for i, s in enumerate(sizes_list):
        if not isinstance(s, (int, np.integer)) or isinstance(s, bool):
            raise ValueError(
                f"training_sizes[{i}] must be an integer, got {type(s).__name__}"
            )
        s_int = int(s)
        if s_int < 1:
            raise ValueError(f"training_sizes[{i}] must be >= 1, got {s_int}")
        if s_int > n_train:
            raise ValueError(
                f"training_sizes[{i}]={s_int} exceeds available train rows ({n_train})"
            )
        cleaned.append(s_int)
    return cleaned
