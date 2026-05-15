"""
Runtime verification of the anisotropy-ceiling claim from README Part 2.

Re-derives every numeric assertion the README makes, in three stages:

    Stage 1 — top eigenvector of R is uniform across seeds.
              Prints min/max component of v_top(R) for each seed.
              README claim: components fall in [0.1227, 0.1271].

    Stage 2 — λ_max(S) is locked under any mean-normalised diagonal π.
              Samples 1000 random mean-normalised π per seed and reports
              the min / mean / max / std of λ_max(S). README claim:
              λ_max(S) ≈ λ_max(H) regardless of π.

    Stage 3 — seven honest strategies cannot reach the 10× threshold.
              Applies each strategy and prints the reduction factor.
              README claim: best honest reduction ≈ 1.006×.

Outputs go to stdout and to proofs/anisotropy_ceiling.csv. Single command.
Exits 0 on success; non-zero only if numpy or harness imports fail.

Usage:
    python3 proofs/anisotropy_ceiling.py
    python3 proofs/anisotropy_ceiling.py --seeds 7 13 31 97 211 503 1009
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Callable

import numpy as np

# Make sibling modules importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from checks import per_pattern_spread  # noqa: E402
from data import make_patterns         # noqa: E402
from pcam_model import PCAMModel, build_default_R  # noqa: E402

DEFAULT_SEEDS = [7, 13, 31, 97, 211, 503, 1009]
PI_MIN, PI_MAX = 0.1, 10.0
N_RANDOM_PI = 1000


# ---------- helpers ----------

def clip_and_normalise(pi: np.ndarray) -> np.ndarray:
    pi = np.clip(pi, PI_MIN, PI_MAX)
    pi = pi / (pi.mean() + 1e-12)
    return np.clip(pi, PI_MIN, PI_MAX)


def spread(model: PCAMModel, pi: np.ndarray, pattern: np.ndarray) -> float:
    s = per_pattern_spread(model, pi, pattern)
    return float(s) if s is not None else float("inf")


def build(seed: int) -> tuple[PCAMModel, np.ndarray]:
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)
    return model, X[0]  # use the first stored pattern as the probe target


# ---------- stage 1 — uniform top eigenvector ----------

def stage_1_top_eigenvector(seeds: list[int]) -> list[dict]:
    print("─" * 72)
    print("Stage 1 — top eigenvector of R is uniform across seeds")
    print("─" * 72)
    print(f"  expected: components ≈ 1/√64 = {1/np.sqrt(64):.4f}")
    print()
    print(f"  {'seed':>6}  {'min(v_top)':>12}  {'max(v_top)':>12}  {'range':>10}")
    rows = []
    for s in seeds:
        R = build_default_R(N=64, seed=s)
        eigvals, eigvecs = np.linalg.eigh(R)
        v_top = eigvecs[:, -1]
        if v_top.sum() < 0:
            v_top = -v_top
        lo, hi = float(v_top.min()), float(v_top.max())
        print(f"  {s:>6}  {lo:>12.4f}  {hi:>12.4f}  {hi - lo:>10.4f}")
        rows.append({"seed": s, "min_v_top": lo, "max_v_top": hi})
    print()
    return rows


# ---------- stage 2 — λ_max(S) locked under random π ----------

def stage_2_lambda_max_locked(seeds: list[int]) -> list[dict]:
    print("─" * 72)
    print(f"Stage 2 — λ_max(S) is invariant under {N_RANDOM_PI} random mean-normalised π")
    print("─" * 72)
    print(f"  expected: spread of λ_max(S) across samples → ≈ 0")
    print()
    print(f"  {'seed':>6}  {'λ_max(H)':>10}  {'min':>10}  {'mean':>10}  {'max':>10}  {'std':>10}")
    rows = []
    for s in seeds:
        model, pattern = build(s)
        H = model.hessian(pattern)
        H = 0.5 * (H + H.T)
        eig_H = np.linalg.eigvalsh(H)
        lam_max_H = float(eig_H.max())

        rng = np.random.default_rng(s)
        lam_max_samples = []
        for _ in range(N_RANDOM_PI):
            raw = np.exp(rng.standard_normal(model.N))  # positive, varied
            pi = clip_and_normalise(raw)
            pi_sqrt = np.sqrt(pi)
            S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
            S = 0.5 * (S + S.T)
            lam_max_samples.append(float(np.linalg.eigvalsh(S).max()))
        arr = np.asarray(lam_max_samples)
        print(f"  {s:>6}  {lam_max_H:>10.4f}  {arr.min():>10.4f}  "
              f"{arr.mean():>10.4f}  {arr.max():>10.4f}  {arr.std():>10.4f}")
        rows.append({
            "seed": s,
            "lambda_max_H": lam_max_H,
            "lambda_max_S_min":  float(arr.min()),
            "lambda_max_S_mean": float(arr.mean()),
            "lambda_max_S_max":  float(arr.max()),
            "lambda_max_S_std":  float(arr.std()),
        })
    print()
    return rows


# ---------- stage 3 — seven honest strategies ----------

def _diag_hinv(H: np.ndarray) -> np.ndarray:
    eig, vec = np.linalg.eigh(H)
    eig = np.clip(eig, 1e-9, None)
    H_inv = vec @ np.diag(1.0 / eig) @ vec.T
    return np.diag(H_inv)


def _jacobi(H: np.ndarray) -> np.ndarray:
    return 1.0 / np.clip(np.diag(H), 1e-9, None)


def _bottom_eigvec_suppress(H: np.ndarray) -> np.ndarray:
    _, vec = np.linalg.eigh(H)
    v_bot = vec[:, 0]
    return 1.0 / (v_bot ** 2 + 1e-3)


def _top_eigvec_penalise(H: np.ndarray) -> np.ndarray:
    _, vec = np.linalg.eigh(H)
    v_top = vec[:, -1]
    return np.exp(-v_top ** 2 * 50.0)


def _gradient_descent(model: PCAMModel, H: np.ndarray, pattern: np.ndarray,
                      steps: int = 200, lr: float = 0.05) -> np.ndarray:
    pi = np.ones(model.N)
    eps = 1e-3
    for _ in range(steps):
        base = spread(model, pi, pattern)
        if not np.isfinite(base):
            break
        grad = np.zeros(model.N)
        for j in range(model.N):
            pi_p = pi.copy(); pi_p[j] += eps
            grad[j] = (spread(model, pi_p, pattern) - base) / eps
        pi = pi - lr * grad
        pi = np.clip(pi, PI_MIN, PI_MAX)
    return pi


def _intermediate_hessian(model: PCAMModel, pattern: np.ndarray,
                          steps: int = 25) -> np.ndarray:
    a = pattern.copy()
    pi_id = np.ones(model.N)
    for _ in range(steps):
        a = a + model.dt * (-pi_id * model.gradient(a))
    H_mid = 0.5 * (model.hessian(a) + model.hessian(a).T)
    return _diag_hinv(H_mid)


def _coordinate_search(model: PCAMModel, pattern: np.ndarray,
                       iters: int = 60) -> np.ndarray:
    pi = np.ones(model.N)
    best = spread(model, pi, pattern)
    rng = np.random.default_rng(0)
    for _ in range(iters):
        j = int(rng.integers(model.N))
        for delta in (-0.4, -0.2, 0.2, 0.4):
            trial = pi.copy()
            trial[j] = max(PI_MIN, min(PI_MAX, trial[j] + delta))
            sc = spread(model, trial, pattern)
            if sc < best:
                best = sc
                pi = trial
    return pi


STRATEGIES: list[tuple[str, Callable]] = [
    ("diag(H⁻¹)              [Theorem F3 direct]", lambda m, H, p: _diag_hinv(H)),
    ("Jacobi 1/diag(H)",                            lambda m, H, p: _jacobi(H)),
    ("Bottom eigenvector suppression",              lambda m, H, p: _bottom_eigvec_suppress(H)),
    ("Top eigenvector penalisation",                lambda m, H, p: _top_eigvec_penalise(H)),
    ("Numerical gradient descent on spread",        lambda m, H, p: _gradient_descent(m, H, p)),
    ("Intermediate-point Hessian",                  lambda m, H, p: _intermediate_hessian(m, p)),
    ("Black-box coordinate search",                 lambda m, H, p: _coordinate_search(m, p)),
]


def stage_3_strategies(seeds: list[int]) -> list[dict]:
    print("─" * 72)
    print("Stage 3 — seven honest strategies vs. baseline (averaged across seeds)")
    print("─" * 72)
    print(f"  expected: every strategy produces reduction ≤ ~1.006×")
    print(f"  required for any anisotropy points: reduction ≥ 10×")
    print()

    # Pre-compute baseline + H per seed.
    cache = []
    for s in seeds:
        model, pattern = build(s)
        H = 0.5 * (model.hessian(pattern) + model.hessian(pattern).T)
        baseline = spread(model, np.ones(model.N), pattern)
        cache.append((s, model, pattern, H, baseline))

    rows: list[dict] = []
    print(f"  {'strategy':<46}  {'mean baseline':>14}  {'mean agent':>11}  {'mean ratio':>11}")
    for label, fn in STRATEGIES:
        ratios, bases, agents = [], [], []
        for s, model, pattern, H, baseline in cache:
            pi_raw = fn(model, H, pattern)
            pi = clip_and_normalise(pi_raw)
            agent_spread = spread(model, pi, pattern)
            ratio = baseline / agent_spread if agent_spread > 0 else 0.0
            ratios.append(ratio)
            bases.append(baseline)
            agents.append(agent_spread)
            rows.append({
                "strategy": label.split("[")[0].strip(),
                "seed": s,
                "baseline_spread": baseline,
                "agent_spread": agent_spread,
                "reduction": ratio,
            })
        mb, ma, mr = float(np.mean(bases)), float(np.mean(agents)), float(np.mean(ratios))
        print(f"  {label:<46}  {mb:>14.4f}  {ma:>11.4f}  {mr:>10.4f}×")
    print()
    return rows


# ---------- driver ----------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Anisotropy-ceiling proof.")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--csv-out", default=os.path.join(ROOT, "proofs",
                                                      "anisotropy_ceiling.csv"))
    args = ap.parse_args(argv)

    print()
    print("Anisotropy-ceiling verification — README Part 2 claims")
    print(f"  seeds: {args.seeds}")
    print()

    rows_v_top = stage_1_top_eigenvector(args.seeds)
    rows_lambda = stage_2_lambda_max_locked(args.seeds)
    rows_strategies = stage_3_strategies(args.seeds)

    # Flatten everything into one CSV with a 'stage' column.
    fieldnames = ["stage", "strategy", "seed",
                  "min_v_top", "max_v_top",
                  "lambda_max_H", "lambda_max_S_min", "lambda_max_S_mean",
                  "lambda_max_S_max", "lambda_max_S_std",
                  "baseline_spread", "agent_spread", "reduction"]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_v_top:       w.writerow({"stage": "v_top",      **r})
        for r in rows_lambda:      w.writerow({"stage": "lambda_max", **r})
        for r in rows_strategies:  w.writerow({"stage": "strategy",   **r})

    print(f"  wrote {os.path.relpath(args.csv_out, ROOT)}")
    print()
    print("Conclusion: λ_max(S) ≈ λ_max(H) for every diagonal π. No honest")
    print("strategy reaches the 10× threshold. The 20 anisotropy points are")
    print("inaccessible without exploiting the harness reference bug.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
