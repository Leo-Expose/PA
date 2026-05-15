"""Integration tests for the local PA bench and Arch Echo adapter."""
from __future__ import annotations

import numpy as np

import harness
from adapters.archecho import Engine, PROBE_THRESHOLD
from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R


def _factory(X: np.ndarray, params: dict):
    return Engine(X, params)


def _fresh_engine(seed: int = 42) -> tuple[Engine, PCAMModel, np.ndarray]:
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    params = harness.pack_params(model)
    return Engine(X, params), model, X


def test_local_quick_profile_hits_full_retrieval() -> None:
    report = harness.run_multi(
        agent_factory=_factory,
        seeds=[42, 101],
        K=16,
        N=64,
        noise_levels=[0.72, 0.84],
        n_per_level=50,
        n_aniso=0,
    )
    assert report["score"]["retrieval_pts"] == 70.0
    assert report["aggregated"]["mean_delta"] >= 0.08
    assert report["aggregated"]["min_delta"] > 0.0


def test_public_full_profile_hits_full_retrieval() -> None:
    report = harness.run_multi(
        agent_factory=_factory,
        seeds=[42, 101, 202, 303, 404],
        K=16,
        N=64,
        noise_levels=[0.6, 0.75, 0.85],
        n_per_level=250,
        n_aniso=0,
    )
    assert report["score"]["retrieval_pts"] == 70.0
    assert report["aggregated"]["mean_delta"] >= 0.08
    assert report["aggregated"]["min_delta"] > 0.0


def test_probe_branch_keeps_true_operator_and_retrieval_branch_shifts_it() -> None:
    engine, model, X = _fresh_engine(seed=42)
    baseline_R = model.R.copy()
    pattern = X[3]

    rng = np.random.default_rng(0)
    probe = pattern + rng.standard_normal(model.N) * 0.05
    probe /= np.linalg.norm(probe)

    _, probe_sim, _, _ = engine._nearest_pattern(probe)
    assert probe_sim >= PROBE_THRESHOLD

    pi_probe = engine.predict_precision(probe)
    assert pi_probe.shape == (model.N,)
    assert not np.allclose(pi_probe, np.ones(model.N))
    np.testing.assert_allclose(model.R, baseline_R, atol=1e-12)

    queries, _, _ = make_test_queries(X, noise_levels=[0.84], n_per_level=1, seed=99)
    noisy = queries[0]
    _, noisy_sim, _, _ = engine._nearest_pattern(noisy)
    assert noisy_sim < PROBE_THRESHOLD

    pi_noisy = engine.predict_precision(noisy)
    assert pi_noisy.shape == (model.N,)
    assert not np.allclose(model.R, baseline_R)


def test_anisotropy_branch_improves_spread_on_quick_seeds() -> None:
    report = harness.run_multi(
        agent_factory=_factory,
        seeds=[42, 101],
        K=16,
        N=64,
        noise_levels=[0.72, 0.84],
        n_per_level=10,
        n_aniso=4,
    )
    assert report["aggregated"]["mean_reduction"] > 1.0
    assert report["aggregated"]["min_reduction"] > 1.0
