"""Tests for the Benchmark class (classical-only, no quantum imports needed)."""
import pytest
import numpy as np


class TestBenchmarkClassicalOnly:
    def test_iris_classical(self):
        from qmc.benchmark import Benchmark
        bench = Benchmark(
            dataset="iris",
            classical_methods=["RF", "LR"],
            quantum_methods=[],  # skip quantum for fast test
        )
        results = bench.run()
        # Benchmark returns structured dict with 'classical' list
        classical_methods = [r["method"] for r in results["classical"]]
        assert "RF" in classical_methods
        assert "LR" in classical_methods
        rf = next(r for r in results["classical"] if r["method"] == "RF")
        assert rf["f1_score"] > 0

    def test_numpy_input(self):
        from qmc.benchmark import Benchmark
        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)
        bench = Benchmark(
            dataset=(X, y),
            classical_methods=["RF"],
            quantum_methods=[],
        )
        results = bench.run()
        classical_methods = [r["method"] for r in results["classical"]]
        assert "RF" in classical_methods

    def test_summary_string(self):
        from qmc.benchmark import Benchmark
        bench = Benchmark(
            dataset="iris",
            classical_methods=["RF"],
            quantum_methods=[],
        )
        bench.run()
        s = bench.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_report_generation(self, tmp_path):
        from qmc.benchmark import Benchmark
        bench = Benchmark(
            dataset="iris",
            classical_methods=["RF"],
            quantum_methods=[],
        )
        bench.run()
        bench.report(str(tmp_path))
        files = list(tmp_path.iterdir())
        assert len(files) >= 1

    def test_dataset_info_in_results(self):
        from qmc.benchmark import Benchmark
        bench = Benchmark(
            dataset="iris",
            classical_methods=["RF"],
            quantum_methods=[],
        )
        results = bench.run()
        assert "dataset_info" in results
        assert results["dataset_info"]["n_features"] == 4
        assert results["dataset_info"]["n_classes"] == 3


class TestDatasetLoading:
    def test_builtin_datasets(self):
        from qmc.datasets.builtin import load_dataset, list_datasets
        names = list_datasets()
        assert "iris" in names
        assert "breast_cancer" in names

    def test_load_iris_tuple(self):
        """load_dataset returns (X_train, X_test, y_train, y_test, meta) tuple."""
        from qmc.datasets.builtin import load_dataset
        result = load_dataset("iris")
        assert isinstance(result, tuple)
        assert len(result) == 5
        X_train, X_test, y_train, y_test, meta = result
        assert X_train.shape[1] == 4
        assert meta.n_classes == 3
        assert meta.name == "iris"

    def test_load_from_arrays(self):
        from qmc.datasets.builtin import load_from_arrays
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        result = load_from_arrays(X, y)
        assert isinstance(result, tuple)
        assert len(result) == 5
        X_train, X_test, y_train, y_test, meta = result
        assert X_train.shape[0] + X_test.shape[0] == 50
        assert X_train.shape[1] == 3

    def test_load_multiple_builtin(self):
        from qmc.datasets.builtin import load_dataset
        for name in ["iris", "breast_cancer", "moons", "circles"]:
            X_train, X_test, y_train, y_test, meta = load_dataset(name)
            assert X_train.shape[0] > 0
            assert X_test.shape[0] > 0
            assert meta.n_features > 0
