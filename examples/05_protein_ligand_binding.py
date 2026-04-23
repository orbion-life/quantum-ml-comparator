"""
Reproduce the +55.6% low-data F1 lift on a 220,471-residue sample of public
protein-ligand-binding data (BioLiP v2).

This example is driven by :class:`qmc.FeatureChannelBenchmark`, which is the
idiomatic qmc API for "same estimator, different feature sets on the same
labels" comparisons. Under the hood the benchmark does the stratified
subsample, the repeated fit / predict / score loop, and the per-channel
lift computation so the script stays short and readable.

What it reproduces (from the dashboard at quantum-binding-explorer):
  - Learning-curve points at 100, 500, 1k, 5k, 10k, 50k training samples
  - The headline "+55.6% at 5,000 samples" result

Runtime: ~3 minutes on a 2021 MacBook Pro (M1).

Data provenance: see benchmarks/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qmc import FeatureChannelBenchmark
from qmc.classical.models import get_random_forest


HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "benchmarks" / "protein_ligand_binding_220k.parquet"

CLASSICAL_COLS = [f"classical_{i:02d}" for i in range(34)]
QUANTUM_COLS = [
    "quantum_0_homo_lumo_gap_eV",
    "quantum_1_occupation_entropy",
    "quantum_2_n_fractional",
    "quantum_3_charge_range",
    "quantum_4_charge_std",
    "quantum_5_n_positive_centers",
    "quantum_6_n_negative_centers",
    "quantum_7_max_deviation_from_integer",
    "quantum_8_total_fractional_occupation",
    "quantum_9_correlation_energy_casci",
]

LEARNING_CURVE_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000]


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Dataset not found at {path}.\n"
            "Expected to find it inside the repository at "
            "benchmarks/protein_ligand_binding_220k.parquet — make sure you "
            "are running the example from a clean checkout."
        )
    return pd.read_parquet(path)


def rf_factory():
    """Fresh RF per channel/size with the hyperparameters the published
    benchmark used (n_estimators=100, max_depth=12, class_weight='balanced')."""
    clf = get_random_forest(n_estimators=100, seed=42)
    clf.set_params(max_depth=12, class_weight="balanced")
    return clf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="include the 154k-sample full-training point (slower)")
    args = parser.parse_args()

    print(f"Loading {DATASET.name} ...")
    df = load(DATASET)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train: {len(tr):,} rows    test: {len(te):,} rows")
    print(f"  label balance (train): "
          f"{tr['label'].value_counts(normalize=True).round(4).to_dict()}")

    X_tr_cls = tr[CLASSICAL_COLS].to_numpy(dtype=np.float32)
    X_tr_all = tr[CLASSICAL_COLS + QUANTUM_COLS].to_numpy(dtype=np.float32)
    X_te_cls = te[CLASSICAL_COLS].to_numpy(dtype=np.float32)
    X_te_all = te[CLASSICAL_COLS + QUANTUM_COLS].to_numpy(dtype=np.float32)
    y_tr = tr["label"].to_numpy(dtype=np.int64)
    y_te = te["label"].to_numpy(dtype=np.int64)

    sizes = LEARNING_CURVE_SIZES + ([len(X_tr_cls)] if args.full else [])

    bench = FeatureChannelBenchmark(
        channels={
            "classical only":      (X_tr_cls, X_te_cls),
            "classical + quantum": (X_tr_all, X_te_all),
        },
        y_train=y_tr,
        y_test=y_te,
        estimator_factory=rf_factory,
        training_sizes=sizes,
        seed=42,
    )

    print("\nTraining and scoring (this takes ~3 min on M1 Pro)…\n")
    bench.run(verbose=False)

    df_out = bench.to_dataframe()
    # Pivot for a clean side-by-side table
    pivot = df_out.pivot(index="n_train", columns="channel", values="score")
    pivot["lift_pct"] = (
        (pivot["classical + quantum"] - pivot["classical only"])
        / pivot["classical only"] * 100
    )
    pivot = pivot.reindex(sizes)

    header = (f"{'n_train':>10}  {'classical F1':>13}  "
              f"{'classical + quantum':>20}  {'lift':>8}")
    print(header)
    print("-" * len(header))
    for n, row in pivot.iterrows():
        print(f"{int(n):>10,}  {row['classical only']:>13.4f}  "
              f"{row['classical + quantum']:>20.4f}  "
              f"{row['lift_pct']:>+7.1f}%")

    out_path = HERE / "05_protein_ligand_binding_results.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    idx_peak = pivot["lift_pct"].idxmax()
    print(f"\nHeadline result: peak lift = {pivot.loc[idx_peak, 'lift_pct']:+.1f}% "
          f"at n_train = {int(idx_peak):,}.")
    print(bench.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
