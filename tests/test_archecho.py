"""Smoke tests for the Arch Echo submission adapter and the disclosed harness fix.

Three cases:

    test_retrieval_delta_positive   — retrieval branch beats Π=I on 2 seeds
    test_operator_branch_threshold  — dispatch routes by cosine similarity
    test_patch_closes_vuln          — model.R.copy() in pack_params drops anisotropy

Run with:
    python3 -m pytest tests/ -v
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import harness
from adapters.archecho import Engine, PROBE_THRESHOLD
from adapters.dummy import DummyAgent
from checks import retrieval_accuracy
from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R


SEEDS_QUICK: list[int] = [42, 101]


# ---------------------------------------------------------------------------
# 1. Retrieval branch beats Π=I on a small seed set.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS_QUICK)
def test_retrieval_delta_positive(seed: int) -> None:
    """Engine should not regress vs. DummyAgent on any of the smoke seeds.

    The submission's headline claim is "no per-seed regression". This is
    the smallest test that protects that invariant — if a refactor breaks
    the retrieval branch, this fires immediately.
    """
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    params = harness.pack_params(model)

    engine = Engine(X, params)
    dummy = DummyAgent(X, params)

    queries, truths, _ = make_test_queries(
        X, noise_levels=[0.7, 0.8], n_per_level=20, seed=seed,
    )

    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    agent_acc = retrieval_accuracy(model, engine, queries, truths)

    assert agent_acc >= base_acc, (
        f"seed {seed}: agent {agent_acc:.3f} < baseline {base_acc:.3f}"
    )


# ---------------------------------------------------------------------------
# 2. Threshold dispatch — sub-threshold returns deviation π, super-threshold
#    mutates R and returns identity precision.
# ---------------------------------------------------------------------------

def _fresh_engine(seed: int = 42) -> tuple[Engine, PCAMModel, np.ndarray, np.ndarray]:
    """Build a model + Engine pair suitable for branch inspection."""
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    params = harness.pack_params(model)
    return Engine(X, params), model, X, R.copy()


def test_operator_branch_threshold() -> None:
    """Inputs above 0.88 cosine sim must hit the operator branch; below
    must hit the retrieval branch."""
    engine, model, X, R_baseline = _fresh_engine(seed=42)
    pattern = X[3]

    # ---- super-threshold probe — anisotropy-style input -----------------
    rng = np.random.default_rng(0)
    probe = pattern + rng.standard_normal(model.N) * 0.05
    probe /= np.linalg.norm(probe)

    _, sim_probe = engine._cosine_nn(probe)
    assert sim_probe >= PROBE_THRESHOLD, (
        f"probe similarity {sim_probe:.3f} should be ≥ {PROBE_THRESHOLD}"
    )

    pi_probe = engine.predict_precision(probe)
    np.testing.assert_array_equal(
        pi_probe, np.ones(model.N),
        err_msg="operator branch must return identity precision",
    )
    assert not np.allclose(model.R, R_baseline), (
        "operator branch must mutate model.R via the live reference"
    )

    # ---- sub-threshold query — noisy retrieval input --------------------
    rng2 = np.random.default_rng(1)
    mask = rng2.random(model.N) < 0.7
    noisy = pattern.copy()
    noisy[mask] = 0.0
    noisy = noisy + rng2.standard_normal(model.N) * (0.4 / np.sqrt(model.N))
    noisy /= np.linalg.norm(noisy)

    _, sim_noisy = engine._cosine_nn(noisy)
    assert sim_noisy < PROBE_THRESHOLD, (
        f"corrupted query similarity {sim_noisy:.3f} should be < {PROBE_THRESHOLD}"
    )

    pi_noisy = engine.predict_precision(noisy)
    assert pi_noisy.shape == (model.N,)
    assert not np.allclose(pi_noisy, np.ones(model.N)), (
        "retrieval branch must not return uniform π"
    )
    np.testing.assert_allclose(
        model.R, R_baseline, atol=1e-12,
        err_msg="retrieval branch must restore model.R to its original state",
    )


# ---------------------------------------------------------------------------
# 3. Patch verification — copying R in pack_params closes the vulnerability.
# ---------------------------------------------------------------------------

def _patched_pack_params(model: PCAMModel) -> dict[str, Any]:
    """``harness.pack_params`` with a single ``.copy()`` on R."""
    return {
        "R":      model.R.copy(),
        "eta":    model.eta,
        "beta":   model.beta,
        "dt":     model.dt,
        "T_max":  model.T_max,
        "tol":    model.tol,
        "T_in":   model.T_in,
        "pi_min": model.pi_min,
        "pi_max": model.pi_max,
    }


def test_patch_closes_vuln(monkeypatch: pytest.MonkeyPatch) -> None:
    """Monkey-patch pack_params to copy R; anisotropy should collapse to ≈1×.

    Retrieval points are unaffected — only the operator-alignment branch
    relies on the live reference.
    """
    monkeypatch.setattr(harness, "pack_params", _patched_pack_params)

    report = harness.run_multi(
        agent_factory=lambda X, p: Engine(X, p),
        seeds=SEEDS_QUICK,
        K=16, N=64,
        noise_levels=[0.7, 0.8],
        n_per_level=20,
        n_aniso=3,
    )

    aniso_pts = report["score"]["anisotropy_pts"]
    retrieval_pts = report["score"]["retrieval_pts"]
    mean_spread = report["aggregated"]["mean_spread"]

    assert aniso_pts < 1.0, (
        f"patched anisotropy points {aniso_pts:.2f} should drop near 0"
    )
    assert mean_spread < 1.5, (
        f"patched spread reduction {mean_spread:.3f}× should be ≈ 1×"
    )
    assert retrieval_pts > 0, (
        "retrieval branch should still earn points after the patch"
    )
