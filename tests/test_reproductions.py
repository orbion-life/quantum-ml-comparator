"""Integration tests for example reproductions.

These execute the real example scripts (not toy versions) and fail
CI if any scientific-sanity gate regresses.

Marked `@pytest.mark.slow` so they don't run by default in fast
pre-commit loops, but they DO run in the full CI suite.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
EXAMPLES = REPO_ROOT / "examples"
FIGURES = EXAMPLES / "figures"


@pytest.mark.slow
def test_anselmetti_h2_reproduction_runs_and_all_gates_pass():
    """Run the Anselmetti et al. H2 reproduction end-to-end.

    The script raises AssertionError on any scientific-sanity gate
    failure (energy vs FCI > 1 mHa, variational violation, minimum
    outside [0.70, 0.78] Å, or VQE-FCI residual discontinuity).
    We treat a non-zero exit as test failure.
    """
    script = EXAMPLES / "reproduce_anselmetti_h2.py"
    assert script.exists(), f"missing script: {script}"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min hard cap; typical runtime on CI is ~1-2 min
    )

    # Script must exit cleanly.
    assert result.returncode == 0, (
        f"reproduce_anselmetti_h2.py exited {result.returncode}.\n"
        f"---STDOUT---\n{result.stdout}\n---STDERR---\n{result.stderr}"
    )

    # The final "ALL GATES PASSED" banner must appear.
    assert "ALL GATES PASSED" in result.stdout, (
        f"Success banner missing from output. Last 2KB:\n"
        f"{result.stdout[-2000:]}"
    )

    # Both output artefacts must be produced.
    png = FIGURES / "h2_dissociation.png"
    csv = FIGURES / "h2_dissociation.csv"
    assert png.exists() and png.stat().st_size > 5_000, f"Figure missing or empty: {png}"
    assert csv.exists() and csv.stat().st_size > 100, f"CSV missing or empty: {csv}"
