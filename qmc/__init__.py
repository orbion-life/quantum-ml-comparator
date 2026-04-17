"""quantum-ml-comparator — compare quantum and classical ML side by side."""

__version__ = "0.2.0"

from qmc.benchmark import Benchmark
from qmc.circuits.sklearn_api import QuantumKernelClassifier, VQCClassifier
from qmc.recommender import (
    get_all_mappings,
    print_recommendations,
    recommend,
)

__all__ = [
    "Benchmark",
    "QuantumKernelClassifier",
    "VQCClassifier",
    "get_all_mappings",
    "print_recommendations",
    "recommend",
]
