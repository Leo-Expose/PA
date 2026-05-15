"""Evaluation primitives for P-04. Pure functions, no orchestration."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pcam_model import PCAMModel


def retrieval_accuracy(model: PCAMModel,
                       agent,
                       queries: np.ndarray,
                       truths: np.ndarray) -> float:
    correct = 0
    for q, t in zip(queries, truths):
        pi = agent.predict_precision(q)
        a_star = model.run(q, pi, u_const=q)
        if model.classify(a_star) == int(t):
            correct += 1
    return correct / len(queries)


def direct_classify_accuracy(model: PCAMModel,
                             queries: np.ndarray,
                             truths: np.ndarray) -> float:
    correct = 0
    for q, t in zip(queries, truths):
        if model.classify(q) == int(t):
            correct += 1
    return correct / len(queries)


def _symmetrised_spread(pi: np.ndarray, H: np.ndarray) -> Optional[float]:
    eig_H = np.linalg.eigvalsh(0.5 * (H + H.T))
    if eig_H.min() <= 0:
        return None
    pi_sqrt = np.sqrt(np.clip(pi, 1e-12, None))
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    if len(eigs) < 2:
        return None
    return float(eigs.max() / eigs.min())


def anisotropy_reductions(model: PCAMModel,
                          agent,
                          pattern_indices: list[int],
                          probe_sigma: float = 0.05,
                          seed: int = 0) -> list[tuple[float, float]]:
    rng = np.random.default_rng(seed)
    pi_I = np.ones(model.N)
    results: list[tuple[float, float]] = []

    for idx in pattern_indices:
        pattern = model.X[idx]
        a_star = model.find_equilibrium(pattern)

        probe = pattern + rng.standard_normal(model.N) * probe_sigma
        probe = probe / max(np.linalg.norm(probe), 1e-12)
        pi_agent_raw = agent.predict_precision(probe)
        pi_agent = model.clip_and_normalise(pi_agent_raw)

        H = model.hessian(a_star)
        s_base = _symmetrised_spread(pi_I, H)
        s_agent = _symmetrised_spread(pi_agent, H)
        if s_base is None or s_agent is None:
            continue
        results.append((s_base, s_agent))

    return results


def summarise_anisotropy(pairs: list[tuple[float, float]]) -> dict[str, float]:
    if not pairs:
        return {
            "baseline_spread": float("nan"),
            "agent_spread": float("nan"),
            "reduction": 0.0,
            "n": 0,
        }
    base = np.array([p[0] for p in pairs])
    agent = np.array([p[1] for p in pairs])
    reductions = base / np.maximum(agent, 1e-12)
    return {
        "baseline_spread": float(np.mean(base)),
        "agent_spread": float(np.mean(agent)),
        "reduction": float(np.mean(reductions)),
        "reduction_min": float(np.min(reductions)),
        "n": len(pairs),
    }
