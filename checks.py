"""Compatibility wrappers around the audited metrics module."""
from __future__ import annotations

import numpy as np

from metrics import (
    anisotropy_reductions,
    direct_classify_accuracy,
    retrieval_accuracy,
    summarise_anisotropy,
)
from pcam_model import PCAMModel


def per_pattern_spread(model: PCAMModel, pi: np.ndarray, pattern: np.ndarray) -> float | None:
    """Compatibility helper used by older local scripts."""
    pi = model.clip_and_normalise(pi)
    equilibrium = model.find_equilibrium(pattern)
    H = model.hessian(equilibrium)
    eig_H = np.linalg.eigvalsh(0.5 * (H + H.T))
    if eig_H.min() <= 0:
        return None
    pi_sqrt = np.sqrt(pi)
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    if len(eigs) < 2:
        return None
    return float(eigs.max() / eigs.min())


def anisotropy_spread(model: PCAMModel,
                      agent,
                      pattern_indices: list[int],
                      probe_sigma: float = 0.05,
                      seed: int = 0) -> float:
    pairs = anisotropy_reductions(model, agent, pattern_indices, probe_sigma=probe_sigma, seed=seed)
    summary = summarise_anisotropy(pairs)
    return float(summary["agent_spread"])


def spread_reduction(model: PCAMModel,
                     agent,
                     baseline,
                     pattern_indices: list[int],
                     seed: int = 0) -> dict[str, float]:
    base = summarise_anisotropy(anisotropy_reductions(model, baseline, pattern_indices, seed=seed))
    yours = summarise_anisotropy(anisotropy_reductions(model, agent, pattern_indices, seed=seed))
    factor = float(base["reduction"] / max(yours["reduction"], 1e-12)) if False else 0.0
    # Preserve the old return shape for any local callers; the harness no longer uses this.
    factor = float(base["agent_spread"] / max(yours["agent_spread"], 1e-12))
    return {
        "baseline_spread": round(base["agent_spread"], 4),
        "agent_spread": round(yours["agent_spread"], 4),
        "reduction_factor": round(factor, 4),
    }
