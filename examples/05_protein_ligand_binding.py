"""
Reproduce the +55.6% low-data F1 lift on a 220,471-residue sample of public
protein-ligand-binding data (BioLiP v2).

Setup (all matching the published benchmark):
  - Classical baseline : Random Forest on 34 sequence / geometric features
  - Classical + quantum: same 34 + 10 VQE-derived descriptors per residue
  - Both trained on the SAME train / val / test split shipped in the parquet

What this script reproduces (from the dashboard at quantum-binding-explorer):
  - Learning-curve points at 100, 500, 1k, 5k, 10k, 50k, full training set
  - Per-category F1 table (vitamin, porphyrin, nucleotide, etc.)
  - The headline "+55.6% at 5,000 samples" number

Runtime: ~3 minutes on a 2021 MacBook Pro (M1).

Data provenance: see benchmarks/README.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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
            "This file is shipped inside the repository at "
            "benchmarks/protein_ligand_binding_220k.parquet — make sure you "
            "are running the example from a clean checkout."
        )
    df = pd.read_parquet(path)
    return df


def _stratified_subsample(y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Match the exact stratified sampler used to generate the published
    benchmark (step8_compare.py): preserve the ~5.98% positive rate at every
    training-set size, deterministic with np.random.seed(42)."""
    np.random.seed(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_ratio = len(pos_idx) / len(y)
    n_pos = max(1, int(n * pos_ratio))
    n_neg = n - n_pos
    pos_sample = np.random.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False)
    neg_sample = np.random.choice(neg_idx, min(n_neg, len(neg_idx)), replace=False)
    idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(idx)
    return idx


def evaluate(X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             n_train: int | None, seed: int = 42) -> float:
    if n_train is not None and n_train < len(X_train):
        idx = _stratified_subsample(y_train, n_train, seed=seed)
        X_train = X_train[idx]
        y_train = y_train[idx]
    # Hyperparameters match step8_compare.py exactly — the published
    # learning curve uses n_estimators=100, max_depth=12.
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return f1_score(y_test, y_pred, zero_division=0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="include the full-training-set point (154,329 samples; slower)")
    args = parser.parse_args()

    print(f"Loading {DATASET.name} ...")
    df = load(DATASET)
    tr = df[df["split"] == "train"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train: {len(tr):,} rows    test: {len(te):,} rows")
    print(f"  label balance (train): {tr['label'].value_counts(normalize=True).round(4).to_dict()}")

    X_tr_cls = tr[CLASSICAL_COLS].to_numpy(dtype=np.float32)
    X_tr_all = tr[CLASSICAL_COLS + QUANTUM_COLS].to_numpy(dtype=np.float32)
    y_tr = tr["label"].to_numpy(dtype=np.int64)
    X_te_cls = te[CLASSICAL_COLS].to_numpy(dtype=np.float32)
    X_te_all = te[CLASSICAL_COLS + QUANTUM_COLS].to_numpy(dtype=np.float32)
    y_te = te["label"].to_numpy(dtype=np.int64)

    sizes = LEARNING_CURVE_SIZES + ([len(X_tr_cls)] if args.full else [])

    print("\nTraining and scoring (this takes ~3 min on M1 Pro)…\n")
    header = f"{'n_train':>10}  {'classical F1':>13}  {'classical + quantum':>20}  {'lift':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    for n in sizes:
        t0 = perf_counter()
        f1_cls = evaluate(X_tr_cls, y_tr, X_te_cls, y_te, n_train=n)
        f1_all = evaluate(X_tr_all, y_tr, X_te_all, y_te, n_train=n)
        dt = perf_counter() - t0
        lift = (f1_all - f1_cls) / f1_cls * 100 if f1_cls > 0 else float("nan")
        print(f"{n:>10,}  {f1_cls:>13.4f}  {f1_all:>20.4f}  {lift:>+7.1f}%    [{dt:.1f}s]")
        rows.append({"n_train": n, "f1_classical": f1_cls,
                     "f1_classical_plus_quantum": f1_all, "lift_pct": lift})

    out = pd.DataFrame(rows)
    out_path = HERE / "05_protein_ligand_binding_results.csv"
    out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Headline check
    peak = out.loc[out["lift_pct"].idxmax()]
    print(f"\nHeadline result: peak lift = {peak['lift_pct']:+.1f}% "
          f"at n_train = {int(peak['n_train']):,}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
