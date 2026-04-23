"""Unit tests for FeatureChannelBenchmark — one estimator across several
feature channels on the same labels."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from qmc import FeatureChannelBenchmark


@pytest.fixture(scope="module")
def synthetic_channels():
    """Two channels on the same labels — second has an extra informative col."""
    rng = np.random.default_rng(0)
    n_tr, n_te = 400, 200
    X_tr = rng.standard_normal((n_tr, 4)).astype(np.float32)
    X_te = rng.standard_normal((n_te, 4)).astype(np.float32)
    # Label is driven by an "extra" feature the baseline channel can't see.
    extra_tr = rng.standard_normal(n_tr).astype(np.float32)
    extra_te = rng.standard_normal(n_te).astype(np.float32)
    y_tr = (extra_tr + 0.3 * X_tr[:, 0] > 0).astype(np.int64)
    y_te = (extra_te + 0.3 * X_te[:, 0] > 0).astype(np.int64)
    return {
        "baseline": (X_tr, X_te),
        "baseline + extra": (
            np.column_stack([X_tr, extra_tr]),
            np.column_stack([X_te, extra_te]),
        ),
        "y_train": y_tr,
        "y_test": y_te,
    }


def _rf_factory():
    # Deep + many trees so the "+extra" channel reliably beats the baseline.
    return RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=0, n_jobs=1)


def test_smoke_runs(synthetic_channels):
    ch = synthetic_channels
    bench = FeatureChannelBenchmark(
        channels={k: ch[k] for k in ("baseline", "baseline + extra")},
        y_train=ch["y_train"], y_test=ch["y_test"],
        estimator_factory=_rf_factory,
        training_sizes=[100, 400], seed=42,
    )
    results = bench.run()
    assert set(results) == {"baseline", "baseline + extra"}
    for channel_name, by_size in results.items():
        assert set(by_size) == {100, 400}
        for record in by_size.values():
            assert 0.0 <= record["score"] <= 1.0
            assert record["fit_seconds"] > 0
            assert record["n_train"] > 0


def test_informative_channel_wins_at_full_size(synthetic_channels):
    ch = synthetic_channels
    bench = FeatureChannelBenchmark(
        channels={k: ch[k] for k in ("baseline", "baseline + extra")},
        y_train=ch["y_train"], y_test=ch["y_test"],
        estimator_factory=_rf_factory,
        training_sizes=[400], seed=42,
    )
    bench.run()
    df = bench.to_dataframe()
    baseline = df[(df["channel"] == "baseline") & (df["n_train"] == 400)]["score"].iloc[0]
    augmented = df[(df["channel"] == "baseline + extra") & (df["n_train"] == 400)]["score"].iloc[0]
    assert augmented > baseline, (
        f"Expected the channel with the extra informative feature to win, "
        f"got baseline={baseline:.3f} augmented={augmented:.3f}"
    )


def test_to_dataframe_has_lift_column(synthetic_channels):
    ch = synthetic_channels
    bench = FeatureChannelBenchmark(
        channels={k: ch[k] for k in ("baseline", "baseline + extra")},
        y_train=ch["y_train"], y_test=ch["y_test"],
        estimator_factory=_rf_factory,
        training_sizes=[200, 400], seed=42,
    )
    bench.run()
    df = bench.to_dataframe()
    # Lift column is named after the FIRST channel (the baseline)
    assert "lift_vs_baseline_pct" in df.columns
    # Baseline rows always have 0% lift vs themselves
    assert (df.loc[df["channel"] == "baseline", "lift_vs_baseline_pct"]
            .abs() < 1e-9).all()
    # Exactly one row per (channel, size)
    assert len(df) == 2 * 2


def test_summary_before_run_raises(synthetic_channels):
    ch = synthetic_channels
    bench = FeatureChannelBenchmark(
        channels={"only": ch["baseline"]},
        y_train=ch["y_train"], y_test=ch["y_test"],
        estimator_factory=_rf_factory,
    )
    with pytest.raises(RuntimeError):
        bench.summary()
    with pytest.raises(RuntimeError):
        bench.to_dataframe()


def test_mismatched_channel_sizes_raises(synthetic_channels):
    ch = synthetic_channels
    X_tr_short = ch["baseline"][0][:10]
    X_te = ch["baseline"][1]
    with pytest.raises(ValueError, match="train/test"):
        FeatureChannelBenchmark(
            channels={"ok": ch["baseline"], "bad": (X_tr_short, X_te)},
            y_train=ch["y_train"], y_test=ch["y_test"],
            estimator_factory=_rf_factory,
        )


def test_mismatched_labels_raises(synthetic_channels):
    ch = synthetic_channels
    with pytest.raises(ValueError, match="y_train"):
        FeatureChannelBenchmark(
            channels={"ok": ch["baseline"]},
            y_train=ch["y_train"][:5], y_test=ch["y_test"],
            estimator_factory=_rf_factory,
        )


def test_custom_scorer(synthetic_channels):
    ch = synthetic_channels
    from sklearn.metrics import accuracy_score

    bench = FeatureChannelBenchmark(
        channels={"only": ch["baseline"]},
        y_train=ch["y_train"], y_test=ch["y_test"],
        estimator_factory=_rf_factory,
        training_sizes=[400], seed=42,
        scorer=accuracy_score,
    )
    results = bench.run()
    score = results["only"][400]["score"]
    # Accuracy on this synthetic task is typically > 0.5 (better than chance)
    assert 0.0 <= score <= 1.0


def test_empty_channels_raises(synthetic_channels):
    ch = synthetic_channels
    with pytest.raises(ValueError):
        FeatureChannelBenchmark(
            channels={},
            y_train=ch["y_train"], y_test=ch["y_test"],
            estimator_factory=_rf_factory,
        )
