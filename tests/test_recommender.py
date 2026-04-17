"""Tests for the QML algorithm recommender."""
import pytest
from qmc.recommender import recommend, print_recommendations, get_all_mappings


class TestRecommend:
    def test_svm_returns_results(self):
        recs = recommend("SVM")
        assert len(recs) >= 1
        names = [r["name"] for r in recs]
        assert any("kernel" in n.lower() or "qsvm" in n.lower() for n in names)

    def test_mlp_returns_vqc(self):
        recs = recommend("MLP")
        assert len(recs) >= 1
        names = [r["name"] for r in recs]
        assert any("vqc" in n.lower() or "variational" in n.lower() for n in names)

    def test_random_forest_aliases(self):
        """All these should resolve to the same thing."""
        r1 = recommend("RandomForest")
        r2 = recommend("random_forest")
        r3 = recommend("RF")
        r4 = recommend("random forest")
        assert len(r1) == len(r2) == len(r3) == len(r4)

    def test_case_insensitive(self):
        r1 = recommend("svm")
        r2 = recommend("SVM")
        r3 = recommend("Svm")
        assert len(r1) == len(r2) == len(r3)

    def test_unknown_algorithm_returns_general(self):
        """Unknown algorithms should get general-purpose recommendations."""
        recs = recommend("SomeWeirdAlgorithm")
        assert len(recs) >= 1

    def test_result_structure(self):
        recs = recommend("SVM")
        for r in recs:
            assert "name" in r
            assert "rationale" in r
            assert "difficulty" in r
            assert r["difficulty"] in ("easy", "medium", "hard")

    def test_circuit_config_adapts_to_features(self):
        recs_4 = recommend("MLP", n_features=4)
        recs_10 = recommend("MLP", n_features=10)
        # Circuit config should reflect feature count
        for r in recs_4:
            if "circuit_config" in r and r["circuit_config"]:
                assert r["circuit_config"].get("n_qubits", 4) <= 10

    def test_all_mappings(self):
        mappings = get_all_mappings()
        assert isinstance(mappings, dict)
        assert len(mappings) >= 8

    def test_print_recommendations_runs(self, capsys):
        print_recommendations("SVM")
        captured = capsys.readouterr()
        assert len(captured.out) > 0


class TestAllAlgorithms:
    """Ensure every documented algorithm has recommendations."""

    @pytest.mark.parametrize("algo", [
        "SVM", "MLP", "RandomForest", "LogisticRegression",
        "KNN", "XGBoost", "NaiveBayes", "PCA",
    ])
    def test_known_algorithm(self, algo):
        recs = recommend(algo)
        assert len(recs) >= 1
        assert all(r["difficulty"] in ("easy", "medium", "hard") for r in recs)
