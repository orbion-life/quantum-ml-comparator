"""
QML Algorithm Recommender: Get quantum counterparts for your ML models.

Tell it what classical algorithm you're using, and it tells you
which quantum algorithms to try and why.
"""
from qmc import recommend, print_recommendations

# Pretty-print recommendations for Random Forest
print("=" * 60)
print("What quantum algorithms should I compare against Random Forest?")
print("=" * 60)
print_recommendations("RandomForest")

print("\n" + "=" * 60)
print("What about SVM?")
print("=" * 60)
print_recommendations("SVM")

print("\n" + "=" * 60)
print("What about XGBoost?")
print("=" * 60)
print_recommendations("XGBoost")

# Programmatic access
print("\n" + "=" * 60)
print("Programmatic API:")
print("=" * 60)
recs = recommend("MLP", n_features=10, n_classes=3)
for r in recs:
    print(f"\n  {r['name']} (difficulty: {r['difficulty']})")
    print(f"    {r['rationale']}")
    print(f"    Circuit config: {r['circuit_config']}")
