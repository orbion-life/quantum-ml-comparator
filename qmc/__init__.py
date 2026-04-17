"""quantum-ml-comparator — compare quantum and classical ML side by side."""

__version__ = "0.1.0"

from qmc.benchmark import Benchmark
from qmc.recommender import recommend, print_recommendations, get_all_mappings

__all__ = [
    "Benchmark",
    "recommend",
    "print_recommendations",
    "get_all_mappings",
]
