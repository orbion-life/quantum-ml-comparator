"""Benchmark orchestrator for quantum vs classical ML comparison.

Three-line API::

    from qmc import Benchmark
    bench = Benchmark(
        dataset="iris",
        quantum_methods=["VQC", "QuantumKernel"],
        classical_methods=["MLP", "SVM", "RF"],
    )
    results = bench.run()
    bench.report("results/")
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from qmc.recommender import recommend
from qmc._validation import (
    BenchmarkConfig,
    FeatureChannelConfig,
    validate_dataset_spec,
    validate_method_list,
    validate_training_sizes,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayPair = Tuple[np.ndarray, np.ndarray]

# ---------------------------------------------------------------------------
# Dataset loaders (no heavy imports at module level)
# ---------------------------------------------------------------------------

_BUILTIN_DATASETS = {
    "iris",
    "breast_cancer",
    "wine",
    "digits",
    "moons",
    "circles",
    "make_classification",
}


def _load_builtin(name: str) -> ArrayPair:
    """Load a built-in dataset by name and return (X, y)."""
    from sklearn import datasets

    name_lower = name.lower().strip()

    if name_lower == "iris":
        data = datasets.load_iris()
    elif name_lower in ("breast_cancer", "breastcancer"):
        data = datasets.load_breast_cancer()
    elif name_lower == "wine":
        data = datasets.load_wine()
    elif name_lower == "digits":
        data = datasets.load_digits()
    elif name_lower == "moons":
        X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)
        return X, y
    elif name_lower == "circles":
        X, y = datasets.make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)
        return X, y
    elif name_lower == "make_classification":
        X, y = datasets.make_classification(
            n_samples=300, n_features=8, n_informative=4,
            n_classes=2, random_state=42,
        )
        return X, y
    else:
        raise ValueError(
            f"Unknown built-in dataset '{name}'. "
            f"Choose from: {sorted(_BUILTIN_DATASETS)}"
        )

    return data.data, data.target


def _load_csv(path: str, target_column: Optional[str] = None) -> ArrayPair:
    """Load a CSV file and return (X, y)."""
    import pandas as pd

    df = pd.read_csv(path)
    if target_column is None:
        # Use the last column as the target
        target_column = df.columns[-1]
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values

    # Encode string targets
    if y.dtype.kind in ("U", "S", "O"):
        y = LabelEncoder().fit_transform(y)

    return X.astype(np.float64), y


# ---------------------------------------------------------------------------
# Classical method dispatcher
# ---------------------------------------------------------------------------

_CLASSICAL_ALIASES: Dict[str, str] = {
    "svm": "SVM",
    "svc": "SVM",
    "mlp": "MLP",
    "neural_network": "MLP",
    "nn": "MLP",
    "rf": "RF",
    "random_forest": "RF",
    "randomforest": "RF",
    "logistic_regression": "LogisticRegression",
    "logreg": "LogisticRegression",
    "lr": "LogisticRegression",
    "knn": "KNN",
    "k_nn": "KNN",
    "xgboost": "XGBoost",
    "xgb": "XGBoost",
    "gradient_boosting": "GradientBoosting",
    "naive_bayes": "NaiveBayes",
    "nb": "NaiveBayes",
    "decision_tree": "DecisionTree",
    "dt": "DecisionTree",
}


def _resolve_classical(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _CLASSICAL_ALIASES.get(key, name)


def _train_classical(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Train a classical model and return a results dict."""
    from sklearn.metrics import accuracy_score, f1_score

    resolved = _resolve_classical(name)

    if resolved == "SVM":
        from sklearn.svm import SVC
        model = SVC(kernel="rbf", random_state=42)
    elif resolved == "MLP":
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
        )
    elif resolved == "RF":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif resolved == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif resolved == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=5)
    elif resolved in ("XGBoost", "GradientBoosting"):
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif resolved == "NaiveBayes":
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
    elif resolved == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Unsupported classical method: {name!r}")

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    return {
        "method": name,
        "type": "classical",
        "accuracy": acc,
        "f1_score": f1,
        "train_time_s": round(train_time, 4),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Quantum method dispatcher (lazy PennyLane imports)
# ---------------------------------------------------------------------------

_QUANTUM_ALIASES: Dict[str, str] = {
    "vqc": "VQC",
    "variational_quantum_classifier": "VQC",
    "quantumkernel": "QuantumKernel",
    "quantum_kernel": "QuantumKernel",
    "qkernel": "QuantumKernel",
    "kernel": "QuantumKernel",
    "quantum_kernel_svm": "QuantumKernel",
}


def _resolve_quantum(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return _QUANTUM_ALIASES.get(key, name)


def _train_quantum(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_qubits: int = 8,
    n_layers: int = 4,
) -> Dict[str, Any]:
    """Train a quantum model and return a results dict.

    Imports PennyLane lazily so the rest of the package stays light.
    """
    import pennylane as qml

    resolved = _resolve_quantum(name)
    n_features = X_train.shape[1]

    # Clamp qubits to feature count
    q = min(n_qubits, n_features)
    dev = qml.device("default.qubit", wires=q)

    if resolved == "VQC":
        return _train_vqc(
            dev, q, n_layers, X_train, y_train, X_test, y_test, name,
        )
    elif resolved == "QuantumKernel":
        return _train_quantum_kernel(
            dev, q, X_train, y_train, X_test, y_test, name,
        )
    else:
        raise ValueError(f"Unsupported quantum method: {name!r}")


# ---- VQC -----------------------------------------------------------------

def _train_vqc(
    dev,
    n_qubits: int,
    n_layers: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method_name: str,
) -> Dict[str, Any]:
    import pennylane as qml
    from sklearn.metrics import accuracy_score, f1_score

    n_features = X_train.shape[1]

    @qml.qnode(dev, interface="numpy")
    def circuit(inputs, weights):
        # Angle-embed features
        for i in range(n_qubits):
            qml.RX(inputs[i % n_features], wires=i)
        # Variational layers
        for layer in range(n_layers):
            for i in range(n_qubits):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
        return qml.expval(qml.PauliZ(0))

    rng = np.random.default_rng(42)
    weights = rng.uniform(-np.pi, np.pi, size=(n_layers, n_qubits, 2))
    step_size = 0.1
    n_epochs = 30

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        for x, y_label in zip(X_train, y_train):
            x_pad = np.zeros(n_features)
            x_pad[: len(x)] = x[:n_features]
            pred = circuit(x_pad, weights)
            # Simple parameter-shift-like gradient step
            grad = np.zeros_like(weights)
            for idx in np.ndindex(weights.shape):
                weights_plus = weights.copy()
                weights_minus = weights.copy()
                weights_plus[idx] += np.pi / 2
                weights_minus[idx] -= np.pi / 2
                grad[idx] = (
                    _vqc_loss(circuit, x_pad, y_label, weights_plus)
                    - _vqc_loss(circuit, x_pad, y_label, weights_minus)
                ) / 2.0
            weights -= step_size * grad
    train_time = time.perf_counter() - t0

    # Predict
    preds = []
    for x in X_test:
        x_pad = np.zeros(n_features)
        x_pad[: len(x)] = x[:n_features]
        val = float(circuit(x_pad, weights))
        preds.append(0 if val >= 0 else 1)

    y_test_bin = (y_test > 0).astype(int) if len(np.unique(y_test)) > 2 else y_test

    acc = float(accuracy_score(y_test_bin, preds))
    f1 = float(f1_score(y_test_bin, preds, average="weighted"))

    return {
        "method": method_name,
        "type": "quantum",
        "accuracy": acc,
        "f1_score": f1,
        "train_time_s": round(train_time, 4),
        "n_qubits": n_qubits,
        "n_layers": n_layers,
    }


def _vqc_loss(circuit_fn, x, y_label, weights):
    pred = float(circuit_fn(x, weights))
    target = 1.0 if y_label == 0 else -1.0
    return (pred - target) ** 2


# ---- Quantum Kernel -------------------------------------------------------

def _train_quantum_kernel(
    dev,
    n_qubits: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method_name: str,
) -> Dict[str, Any]:
    import pennylane as qml
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score

    n_features = X_train.shape[1]

    @qml.qnode(dev, interface="numpy")
    def kernel_circuit(x1, x2):
        # Encode x1
        for i in range(n_qubits):
            qml.RX(x1[i % n_features], wires=i)
            qml.RZ(x1[i % n_features], wires=i)
        # Adjoint of x2 encoding
        for i in reversed(range(n_qubits)):
            qml.adjoint(qml.RZ)(x2[i % n_features], wires=i)
            qml.adjoint(qml.RX)(x2[i % n_features], wires=i)
        return qml.probs(wires=range(n_qubits))

    def kernel_value(x1, x2):
        probs = kernel_circuit(x1, x2)
        return float(probs[0])  # overlap = probability of |00...0>

    t0 = time.perf_counter()

    # Build kernel matrices
    n_train = len(X_train)
    n_test = len(X_test)

    K_train = np.zeros((n_train, n_train))
    for i in range(n_train):
        for j in range(i, n_train):
            val = kernel_value(X_train[i], X_train[j])
            K_train[i, j] = val
            K_train[j, i] = val

    K_test = np.zeros((n_test, n_train))
    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = kernel_value(X_test[i], X_train[j])

    svm = SVC(kernel="precomputed", random_state=42)
    svm.fit(K_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = svm.predict(K_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted"))

    return {
        "method": method_name,
        "type": "quantum",
        "accuracy": acc,
        "f1_score": f1,
        "train_time_s": round(train_time, 4),
        "n_qubits": n_qubits,
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def _compare(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build a sorted comparison table from individual results."""
    table = []
    for r in results:
        table.append({
            "method": r["method"],
            "type": r["type"],
            "accuracy": r["accuracy"],
            "f1_score": r["f1_score"],
            "train_time_s": r["train_time_s"],
        })
    table.sort(key=lambda x: x["accuracy"], reverse=True)
    return table


# ---------------------------------------------------------------------------
# Benchmark class
# ---------------------------------------------------------------------------

class Benchmark:
    """Orchestrator for quantum vs classical ML benchmarks.

    Parameters
    ----------
    dataset:
        One of: built-in name (``"iris"``, ``"breast_cancer"``, ``"moons"``,
        ``"circles"``, etc.), a path to a CSV file, or an ``(X, y)`` tuple
        of numpy arrays.
    target_column:
        Column name to use as target when *dataset* is a CSV path.
    quantum_methods:
        List of quantum method names (e.g. ``["VQC", "QuantumKernel"]``).
        If *None* and *classical_methods* is provided, methods are
        auto-recommended via :func:`qmc.recommender.recommend`.
    classical_methods:
        List of classical method names (e.g. ``["MLP", "SVM", "RF"]``).
    n_qubits:
        Number of qubits for quantum circuits.
    n_layers:
        Number of variational layers for VQC circuits.
    test_size:
        Fraction of data held out for testing.
    random_state:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Union[str, ArrayPair] = "iris",
        target_column: Optional[str] = None,
        quantum_methods: Optional[Sequence[str]] = None,
        classical_methods: Optional[Sequence[str]] = None,
        n_qubits: int = 8,
        n_layers: int = 4,
        test_size: float = 0.3,
        random_state: int = 42,
    ) -> None:
        # --- Sanitize / validate inputs at the trust boundary ---
        cfg = BenchmarkConfig(
            target_column=target_column,
            n_qubits=n_qubits,
            n_layers=n_layers,
            test_size=test_size,
            random_state=random_state,
        )
        validated_dataset = validate_dataset_spec(dataset)
        validated_classical = validate_method_list("classical_methods", classical_methods)
        validated_quantum = validate_method_list("quantum_methods", quantum_methods)

        self.dataset_spec = validated_dataset
        self.target_column = cfg.target_column
        self.n_qubits = cfg.n_qubits
        self.n_layers = cfg.n_layers
        self.test_size = cfg.test_size
        self.random_state = cfg.random_state

        # Resolve method lists
        self.classical_methods: List[str] = (
            validated_classical if validated_classical is not None
            else ["MLP", "SVM", "RF"]
        )

        if validated_quantum is not None:
            self.quantum_methods: List[str] = validated_quantum
        elif validated_classical is not None:
            # Auto-recommend from classical methods
            self.quantum_methods = self._auto_recommend(self.classical_methods)
        else:
            self.quantum_methods = ["VQC", "QuantumKernel"]

        # Filled after run()
        self._results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_recommend(classical_methods: List[str]) -> List[str]:
        """Use the recommender to pick quantum methods for each classical one."""
        seen: set[str] = set()
        methods: List[str] = []
        for cm in classical_methods:
            recs = recommend(cm)
            for rec in recs:
                name = rec["name"]
                # Map recommendation names to our dispatcher keys
                dispatcher_name = _resolve_quantum(name)
                if dispatcher_name not in seen:
                    seen.add(dispatcher_name)
                    methods.append(dispatcher_name)
        # Ensure at least VQC and QuantumKernel
        for default in ("VQC", "QuantumKernel"):
            if default not in seen:
                methods.append(default)
        return methods

    def _load_dataset(self) -> ArrayPair:
        """Load data from whatever format the user supplied."""
        ds = self.dataset_spec

        if isinstance(ds, tuple):
            X, y = ds
            return np.asarray(X, dtype=np.float64), np.asarray(y)

        if isinstance(ds, str):
            # Check if it is a file path
            if os.path.isfile(ds):
                return _load_csv(ds, self.target_column)
            # Otherwise treat as built-in name
            return _load_builtin(ds)

        raise TypeError(
            f"dataset must be a string name, a file path, or an (X, y) tuple, "
            f"got {type(ds).__name__}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        learning_curve_sizes: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Train all methods and return a results dictionary.

        Parameters
        ----------
        learning_curve_sizes:
            Optional list of training-set fractions (e.g. ``[0.2, 0.5, 1.0]``)
            to generate learning curves.  If *None*, a single full run is
            performed.

        Returns
        -------
        dict
            Keys: ``"classical"``, ``"quantum"``, ``"comparison"``,
            ``"dataset_info"``, and optionally ``"learning_curves"``.
        """
        X, y = self._load_dataset()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None,
        )

        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))

        dataset_info = {
            "n_samples": len(X),
            "n_features": n_features,
            "n_classes": n_classes,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }
        if isinstance(self.dataset_spec, str):
            dataset_info["name"] = self.dataset_spec

        # --- Train classical ---
        classical_results: List[Dict[str, Any]] = []
        for cm in self.classical_methods:
            try:
                res = _train_classical(cm, X_train, y_train, X_test, y_test)
                classical_results.append(res)
            except Exception as exc:
                classical_results.append({
                    "method": cm,
                    "type": "classical",
                    "accuracy": None,
                    "f1_score": None,
                    "train_time_s": None,
                    "error": str(exc),
                })

        # --- Train quantum ---
        quantum_results: List[Dict[str, Any]] = []
        for qm in self.quantum_methods:
            try:
                res = _train_quantum(
                    qm, X_train, y_train, X_test, y_test,
                    n_qubits=self.n_qubits, n_layers=self.n_layers,
                )
                quantum_results.append(res)
            except Exception as exc:
                quantum_results.append({
                    "method": qm,
                    "type": "quantum",
                    "accuracy": None,
                    "f1_score": None,
                    "train_time_s": None,
                    "error": str(exc),
                })

        # --- Comparison ---
        all_ok = [
            r for r in classical_results + quantum_results
            if r.get("accuracy") is not None
        ]
        comparison = _compare(all_ok)

        # --- Learning curves (optional) ---
        learning_curves = None
        if learning_curve_sizes is not None:
            learning_curves = self._learning_curves(
                learning_curve_sizes, X_scaled, y, n_features,
            )

        self._results = {
            "dataset_info": dataset_info,
            "classical": classical_results,
            "quantum": quantum_results,
            "comparison": comparison,
        }
        if learning_curves is not None:
            self._results["learning_curves"] = learning_curves

        return self._results

    def _learning_curves(
        self,
        sizes: Sequence[float],
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run each method at multiple training-set fractions."""
        curves: Dict[str, List[Dict[str, Any]]] = {}

        for frac in sizes:
            n_use = max(10, int(len(X) * frac))
            idx = np.random.default_rng(self.random_state).choice(
                len(X), size=n_use, replace=False,
            )
            X_sub, y_sub = X[idx], y[idx]
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_sub, y_sub, test_size=self.test_size,
                random_state=self.random_state,
            )

            for cm in self.classical_methods:
                try:
                    res = _train_classical(cm, X_tr, y_tr, X_te, y_te)
                    res["fraction"] = frac
                    curves.setdefault(cm, []).append(res)
                except Exception:
                    pass

            for qm in self.quantum_methods:
                try:
                    res = _train_quantum(
                        qm, X_tr, y_tr, X_te, y_te,
                        n_qubits=self.n_qubits, n_layers=self.n_layers,
                    )
                    res["fraction"] = frac
                    curves.setdefault(qm, []).append(res)
                except Exception:
                    pass

        return curves

    def report(self, output_dir: str) -> str:
        """Generate a Markdown report and write it to *output_dir*.

        Returns
        -------
        str
            Path to the generated report file.
        """
        if self._results is None:
            raise RuntimeError("Call .run() before .report()")

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "benchmark_report.md"

        lines: List[str] = []
        info = self._results["dataset_info"]
        ds_name = info.get("name", "custom")

        lines.append(f"# Quantum vs Classical ML Benchmark: {ds_name}")
        lines.append("")
        lines.append("## Dataset")
        lines.append("")
        lines.append(f"- Samples: {info['n_samples']}")
        lines.append(f"- Features: {info['n_features']}")
        lines.append(f"- Classes: {info['n_classes']}")
        lines.append(f"- Train/Test split: {info['train_size']}/{info['test_size']}")
        lines.append("")

        # --- Comparison table ---
        lines.append("## Results")
        lines.append("")
        lines.append("| Rank | Method | Type | Accuracy | F1 Score | Train Time (s) |")
        lines.append("|------|--------|------|----------|----------|----------------|")
        for rank, row in enumerate(self._results["comparison"], 1):
            acc = f"{row['accuracy']:.4f}" if row["accuracy"] is not None else "N/A"
            f1 = f"{row['f1_score']:.4f}" if row["f1_score"] is not None else "N/A"
            tt = f"{row['train_time_s']:.4f}" if row["train_time_s"] is not None else "N/A"
            lines.append(f"| {rank} | {row['method']} | {row['type']} | {acc} | {f1} | {tt} |")
        lines.append("")

        # --- Classical details ---
        lines.append("## Classical Methods")
        lines.append("")
        for r in self._results["classical"]:
            if r.get("error"):
                lines.append(f"- **{r['method']}**: ERROR - {r['error']}")
            else:
                lines.append(
                    f"- **{r['method']}**: accuracy={r['accuracy']:.4f}, "
                    f"f1={r['f1_score']:.4f}, time={r['train_time_s']:.4f}s"
                )
        lines.append("")

        # --- Quantum details ---
        lines.append("## Quantum Methods")
        lines.append("")
        for r in self._results["quantum"]:
            if r.get("error"):
                lines.append(f"- **{r['method']}**: ERROR - {r['error']}")
            else:
                extra = ""
                if "n_qubits" in r:
                    extra += f", qubits={r['n_qubits']}"
                if "n_layers" in r:
                    extra += f", layers={r['n_layers']}"
                lines.append(
                    f"- **{r['method']}**: accuracy={r['accuracy']:.4f}, "
                    f"f1={r['f1_score']:.4f}, time={r['train_time_s']:.4f}s{extra}"
                )
        lines.append("")

        # --- Errors ---
        errors = [
            r for r in self._results["classical"] + self._results["quantum"]
            if r.get("error")
        ]
        if errors:
            lines.append("## Errors")
            lines.append("")
            for r in errors:
                lines.append(f"- {r['method']} ({r['type']}): {r['error']}")
            lines.append("")

        content = "\n".join(lines)
        report_path.write_text(content, encoding="utf-8")
        return str(report_path)

    def summary(self) -> str:
        """Return a one-line summary of benchmark results."""
        if self._results is None:
            raise RuntimeError("Call .run() before .summary()")

        comp = self._results["comparison"]
        if not comp:
            return "No methods completed successfully."

        best = comp[0]
        n_classical = sum(1 for r in comp if r["type"] == "classical")
        n_quantum = sum(1 for r in comp if r["type"] == "quantum")
        ds_name = self._results["dataset_info"].get("name", "custom dataset")

        return (
            f"Best: {best['method']} ({best['type']}) with "
            f"{best['accuracy']:.4f} accuracy on {ds_name} "
            f"({n_classical} classical, {n_quantum} quantum methods compared)"
        )


# ---------------------------------------------------------------------------
# FeatureChannelBenchmark — compare the same estimator across different
# feature sets on the same labels. Useful when the experimental question is
# "does adding this feature channel improve the model?" rather than "which
# model wins on this dataset?".
# ---------------------------------------------------------------------------

from typing import Callable, Mapping  # noqa: E402


class FeatureChannelBenchmark:
    """Compare one estimator against multiple feature channels on the same labels.

    Where :class:`Benchmark` compares *different methods on one dataset*, this
    class compares *one method on different feature sets*. The canonical use
    case is adding a new feature channel (e.g. quantum-chemistry descriptors)
    to an existing classical feature set and measuring the lift.

    Parameters
    ----------
    channels:
        Mapping from human-readable channel name (e.g. ``"classical only"``)
        to a tuple ``(X_train, X_test)`` of feature arrays. All channels must
        share the same labels and the same number of train / test rows.
    y_train, y_test:
        Labels, shared across channels.
    estimator_factory:
        Zero-argument callable returning an unfitted estimator (a fresh copy
        per channel and per training size, so that subsequent .fit() calls
        don't share state). Must satisfy the sklearn ``fit``/``predict``
        contract; PyTorch models from :mod:`qmc.classical.models` also work.
    training_sizes:
        Optional list of training-set sizes to sweep. When ``None`` a single
        run is performed at the full training-set size. Each size is drawn
        from ``X_train`` by stratified subsampling (when ``stratified=True``)
        or uniform random sampling.
    stratified:
        If ``True`` (default) subsampling preserves the label distribution.
    seed:
        Random seed for the subsampler (default 42).
    scorer:
        Optional callable ``(y_true, y_pred) -> float`` that returns a
        single summary score per fit. Defaults to binary F1 on the
        positive class (class 1), matching the published benchmark. Pass
        any scalar-returning scorer to use a different target metric.

    Examples
    --------
    >>> from qmc import FeatureChannelBenchmark
    >>> from qmc.classical.models import get_random_forest
    >>> bench = FeatureChannelBenchmark(
    ...     channels={
    ...         "classical only":      (X_tr_cls, X_te_cls),
    ...         "classical + quantum": (X_tr_all, X_te_all),
    ...     },
    ...     y_train=y_tr, y_test=y_te,
    ...     estimator_factory=lambda: get_random_forest(n_estimators=100, seed=42),
    ...     training_sizes=[100, 500, 1000, 5000, 10000, 50000],
    ... )
    >>> results = bench.run()
    >>> bench.summary()                   # one-line "best channel" string
    >>> bench.to_dataframe()               # long-form DataFrame with lifts
    """

    def __init__(
        self,
        channels: Mapping[str, ArrayPair],
        y_train: np.ndarray,
        y_test: np.ndarray,
        estimator_factory: Callable[[], Any],
        training_sizes: Optional[Sequence[int]] = None,
        stratified: bool = True,
        seed: int = 42,
        scorer: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> None:
        # --- Scalar config goes through pydantic ---
        cfg = FeatureChannelConfig(stratified=stratified, seed=seed)

        # --- Channel/array shape validation (same checks, kept here as
        # they are about *cross-argument* dimensional consistency, which
        # pydantic isn't the right tool for) ---
        if not channels:
            raise ValueError("channels must contain at least one entry")
        first_key = next(iter(channels))
        X_tr_ref, X_te_ref = channels[first_key]
        n_tr, n_te = len(X_tr_ref), len(X_te_ref)
        for name, (X_tr, X_te) in channels.items():
            if len(X_tr) != n_tr or len(X_te) != n_te:
                raise ValueError(
                    f"Channel '{name}' has {len(X_tr)}/{len(X_te)} train/test "
                    f"rows, expected {n_tr}/{n_te} to match the other channels."
                )
        if len(y_train) != n_tr:
            raise ValueError(f"y_train has {len(y_train)} rows, expected {n_tr}.")
        if len(y_test) != n_te:
            raise ValueError(f"y_test has {len(y_test)} rows, expected {n_te}.")

        # estimator_factory must be callable; sklearn duck-typed beyond that.
        if not callable(estimator_factory):
            raise TypeError(
                f"estimator_factory must be callable, got {type(estimator_factory).__name__}"
            )

        self.channels: Dict[str, ArrayPair] = dict(channels)
        self.y_train = np.asarray(y_train)
        self.y_test = np.asarray(y_test)
        self.estimator_factory = estimator_factory
        validated_sizes = validate_training_sizes(training_sizes, n_tr)
        self.training_sizes: List[int] = (
            validated_sizes if validated_sizes is not None else [n_tr]
        )
        self.stratified = cfg.stratified
        self.seed = cfg.seed

        if scorer is None:
            from sklearn.metrics import f1_score

            def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

            self.scorer: Callable[[np.ndarray, np.ndarray], float] = _binary_f1
        else:
            self.scorer = scorer

        self._results: Optional[Dict[str, Dict[int, Dict[str, float]]]] = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _subsample(self, y: np.ndarray, n: int) -> np.ndarray:
        """Deterministic train-index sampler. Stratified when enabled, else uniform."""
        rng_state = np.random.get_state()
        try:
            np.random.seed(self.seed)
            if not self.stratified:
                idx = np.random.choice(len(y), size=min(n, len(y)), replace=False)
                return np.asarray(idx)
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            pos_ratio = len(pos_idx) / len(y) if len(y) > 0 else 0.0
            n_pos = max(1, int(n * pos_ratio))
            n_neg = n - n_pos
            pos_sample = np.random.choice(
                pos_idx, min(n_pos, len(pos_idx)), replace=False)
            neg_sample = np.random.choice(
                neg_idx, min(n_neg, len(neg_idx)), replace=False)
            idx = np.concatenate([pos_sample, neg_sample])
            np.random.shuffle(idx)
            return idx
        finally:
            np.random.set_state(rng_state)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, verbose: bool = False) -> Dict[str, Dict[int, Dict[str, float]]]:
        """Train and score the estimator on every (channel, size) combination."""
        results: Dict[str, Dict[int, Dict[str, float]]] = {
            name: {} for name in self.channels
        }
        for size in self.training_sizes:
            idx = self._subsample(self.y_train, size)
            y_tr_sub = self.y_train[idx]
            for name, (X_tr, X_te) in self.channels.items():
                X_tr_sub = X_tr[idx]
                clf = self.estimator_factory()
                t0 = time.perf_counter()
                clf.fit(X_tr_sub, y_tr_sub)
                y_pred = clf.predict(X_te)
                dt = time.perf_counter() - t0
                score = self.scorer(self.y_test, y_pred)
                results[name][size] = {"score": score, "fit_seconds": dt,
                                       "n_train": len(idx)}
                if verbose:
                    print(f"  {name:>22s} @ n={size:>7d}: "
                          f"score={score:.4f}  [{dt:.1f}s]")
        self._results = results
        return results

    def to_dataframe(self):  # type: ignore[no-untyped-def]
        """Return a long-format DataFrame: channel, n_train, score, lift_vs_first."""
        if self._results is None:
            raise RuntimeError("Call .run() before .to_dataframe()")
        import pandas as pd

        first_channel = next(iter(self.channels))
        rows: List[Dict[str, Any]] = []
        for channel, by_size in self._results.items():
            for size, record in by_size.items():
                baseline = self._results[first_channel][size]["score"]
                score = record["score"]
                lift = (score - baseline) / baseline * 100 if baseline > 0 else float("nan")
                rows.append({
                    "channel": channel,
                    "n_train": size,
                    "score": score,
                    f"lift_vs_{first_channel}_pct": lift,
                    "fit_seconds": record["fit_seconds"],
                })
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """One-line "best channel" summary using the largest training size."""
        if self._results is None:
            raise RuntimeError("Call .run() before .summary()")
        last_size = max(self.training_sizes)
        ranking = sorted(
            self._results.items(),
            key=lambda kv: kv[1][last_size]["score"],
            reverse=True,
        )
        best_name, best_scores = ranking[0]
        return (
            f"Best channel at n_train={last_size:,}: "
            f"{best_name} (score={best_scores[last_size]['score']:.4f}) "
            f"across {len(self.channels)} feature channel(s) "
            f"and {len(self.training_sizes)} training-size(s)."
        )
