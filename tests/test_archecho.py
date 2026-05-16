"""
pytest suite for the Arch Echo adapter.

Three tests, matching the README's test table:

    test_retrieval_delta_positive
        Engine beats DummyAgent on seeds 42 and 101.

    test_anisotropy_reduction_positive
        Mean spread reduction > 1.0× across patterns on seed 42.

    test_probe_dispatch
        High-similarity inputs (sim ≥ 0.89) return the anisotropy π;
        low-similarity inputs return the retrieval π.
        Verified by checking that the two π vectors are meaningfully
        different and that the high-sim path matches the cached
        anisotropy bank entry for the nearest pattern.
"""
from __future__ import annotations

import numpy as np
import pytest

from adapters.archecho import Engine
from adapters.dummy import DummyAgent
from data import make_patterns, make_test_queries
from metrics import anisotropy_reductions, retrieval_accuracy, summarise_anisotropy
from pcam_model import PCAMModel, build_default_R


# ---------- shared fixtures ----------

def _build(seed: int = 42) -> tuple[PCAMModel, np.ndarray, dict]:
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    params = {
        "R": model.R,
        "eta": model.eta,
        "beta": model.beta,
        "dt": model.dt,
        "T_max": model.T_max,
        "tol": model.tol,
        "T_in": model.T_in,
        "pi_min": model.pi_min,
        "pi_max": model.pi_max,
    }
    return model, X, params


# ---------- test 1 — retrieval delta positive ----------

@pytest.mark.parametrize("seed", [42, 101])
def test_retrieval_delta_positive(seed: int) -> None:
    """Engine accuracy must exceed DummyAgent (Π=I) on seeds 42 and 101."""
    model, X, params = _build(seed)
    queries, truths, _ = make_test_queries(
        X,
        noise_levels=[0.6, 0.75, 0.85],
        n_per_level=40,   # light but representative
        seed=seed,
    )

    engine = Engine(X, params)
    dummy = DummyAgent(X, params)

    engine_acc = retrieval_accuracy(model, engine, queries, truths)
    dummy_acc = retrieval_accuracy(model, dummy, queries, truths)

    delta = engine_acc - dummy_acc
    assert delta > 0, (
        f"seed={seed}: Engine ({engine_acc:.3f}) did not beat DummyAgent "
        f"({dummy_acc:.3f}), Δ={delta:+.3f}"
    )


# ---------- test 2 — anisotropy reduction positive ----------

def test_anisotropy_reduction_positive() -> None:
    """Mean spread reduction must be > 1.0× across all 16 patterns on seed 42."""
    model, X, params = _build(42)
    engine = Engine(X, params)

    pairs = anisotropy_reductions(model, engine, list(range(len(X))), seed=42)
    summary = summarise_anisotropy(pairs)

    assert summary["reduction"] > 1.0, (
        f"Mean spread reduction {summary['reduction']:.4f}× is not > 1.0×"
    )
    assert summary["n"] > 0, "No valid anisotropy pairs were computed"


# ---------- test 3 — probe dispatch ----------

def test_probe_dispatch() -> None:
    """
    High-similarity queries (sim ≥ 0.89) must route to the anisotropy branch;
    low-similarity queries must route to the retrieval branch.

    Verified by:
      - high-sim path: π matches the cached anisotropy bank for the nearest
        pattern (cosine similarity > 0.99 with the bank entry).
      - low-sim path: π is meaningfully different from the anisotropy bank
        entry (cosine similarity < 0.99).
      - the two π vectors are not identical to each other.
    """
    model, X, params = _build(42)
    engine = Engine(X, params)

    rng = np.random.default_rng(0)

    # High-sim query: take a stored pattern and add tiny noise so sim ≥ 0.89.
    pattern_idx = 0
    clean = X[pattern_idx].copy()
    noise = rng.standard_normal(64) * 0.02
    high_sim_query = clean + noise
    high_sim_query /= np.linalg.norm(high_sim_query)

    # Confirm the query actually lands above the threshold.
    sims = X @ high_sim_query
    assert sims.max() >= 0.89, (
        f"Test setup failed: max sim {sims.max():.3f} < 0.89 — "
        "increase noise level or pick a different pattern"
    )

    # Low-sim query: heavy corruption so sim < 0.89.
    low_sim_query = rng.standard_normal(64)
    low_sim_query /= np.linalg.norm(low_sim_query)
    # Verify it's actually low-sim.
    assert (X @ low_sim_query).max() < 0.89, (
        "Test setup failed: random query unexpectedly has sim ≥ 0.89"
    )

    pi_high = engine.predict_precision(high_sim_query)
    pi_low = engine.predict_precision(low_sim_query)

    # High-sim π should closely match the anisotropy bank for the nearest pattern.
    bank_entry = engine._anisotropy_bank[pattern_idx]
    cos_high = float(
        np.dot(pi_high, bank_entry)
        / (np.linalg.norm(pi_high) * np.linalg.norm(bank_entry) + 1e-12)
    )
    assert cos_high > 0.99, (
        f"High-sim π (cos={cos_high:.4f}) does not match anisotropy bank — "
        "dispatch may not be routing to the anisotropy branch"
    )

    # Low-sim π should differ from the anisotropy bank.
    cos_low = float(
        np.dot(pi_low, bank_entry)
        / (np.linalg.norm(pi_low) * np.linalg.norm(bank_entry) + 1e-12)
    )
    assert cos_low < 0.99, (
        f"Low-sim π (cos={cos_low:.4f}) is too similar to anisotropy bank — "
        "dispatch may not be routing to the retrieval branch"
    )

    # The two π vectors must not be identical.
    assert not np.allclose(pi_high, pi_low), (
        "High-sim and low-sim π are identical — dispatch is not branching"
    )
