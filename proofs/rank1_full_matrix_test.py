"""
Rank-1 precision diagnostic — is the diagonal constraint THE bottleneck?

A full-matrix preconditioner Π = I + α·vvᵀ cannot be submitted through the
bench (the submission interface is diagonal-only). This script is a pure
diagnostic that asks: if we lift the diagonal constraint and allow rank-1
updates, can the spread of Π^½ H Π^½ break the ~1.3× ceiling that the best
diagonal π hits?

Three rank-1 strategies per pattern, compared against the best diagonal π:

    1. Boost bottom eigenvector:   Π = I + α · v_min v_minᵀ   (lift λ_min)
    2. Suppress top eigenvector:   Π = I − α · v_max v_maxᵀ   (push λ_max down)
    3. Both simultaneously:        Π = I + α · v_min v_minᵀ − β · v_max v_maxᵀ

Interpretation:

    • If (1) or (2) reduces spread far more than the diagonal best, the
      diagonal projection is the binding constraint — full-matrix Π would
      help if the interface ever allowed it.
    • If all three sit close to the diagonal best, the H structure itself
      is fundamentally ill-conditioned and no linear preconditioner (even
      a full matrix one) can fix it. The 1.3× ceiling is structural, not
      a quirk of the diagonal-only API.

Outputs go to stdout and to proofs/diag_rank1.csv.

Usage:
    python3 proofs/diag_rank1.py
    python3 proofs/diag_rank1.py --seeds 42 101 202
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

# Make sibling modules importable when run from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data import make_patterns                         # noqa: E402
from metrics import _symmetrised_spread                # noqa: E402
from pcam_model import PCAMModel, build_default_R     # noqa: E402

DEFAULT_SEEDS = [42, 101, 202]
ALPHA_BOOST = np.linspace(0.01, 20.0, 200)
ALPHA_SUPPRESS = np.linspace(0.01, 20.0, 200)
DOUBLE_ALPHA = np.linspace(0.01, 15.0, 100)
DOUBLE_BETA = np.linspace(0.01, 0.95, 100)


# ---------- helpers ----------

def full_matrix_spread(H: np.ndarray, Pi_full: np.ndarray) -> float:
    """Spread of Π^½ H Π^½ for an arbitrary symmetric PSD Π."""
    eig_pi, V = np.linalg.eigh(Pi_full)
    eig_pi = np.clip(eig_pi, 1e-12, None)
    Pi_sqrt = V @ np.diag(np.sqrt(eig_pi)) @ V.T
    S = Pi_sqrt @ H @ Pi_sqrt
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    if len(eigs) < 2:
        return float("inf")
    return float(eigs[-1] / eigs[0])


def optimize_rank1(H: np.ndarray,
                   v: np.ndarray,
                   mode: str = "boost",
                   alpha_range: np.ndarray = ALPHA_BOOST) -> tuple[float, float]:
    """Sweep α for Π = I ± α·vvᵀ and return (best_alpha, best_spread)."""
    best_spread = float("inf")
    best_alpha = 0.0
    N = len(v)
    I = np.eye(N)
    vvT = np.outer(v, v)
    for alpha in alpha_range:
        if mode == "boost":
            Pi = I + alpha * vvT
        else:  # suppress
            if alpha >= 0.99:
                continue
            Pi = I - alpha * vvT
        if np.linalg.eigvalsh(Pi).min() <= 0:
            continue
        s = full_matrix_spread(H, Pi)
        if s < best_spread:
            best_spread = s
            best_alpha = float(alpha)
    return best_alpha, best_spread


def optimize_double_rank1(H: np.ndarray,
                          v_min: np.ndarray,
                          v_max: np.ndarray,
                          alpha_range: np.ndarray = DOUBLE_ALPHA,
                          beta_range: np.ndarray = DOUBLE_BETA
                          ) -> tuple[tuple[float, float], float]:
    """Sweep (α, β) for Π = I + α·v_min v_minᵀ − β·v_max v_maxᵀ."""
    best_spread = float("inf")
    best_ab = (0.0, 0.0)
    N = len(v_min)
    I = np.eye(N)
    A = np.outer(v_min, v_min)
    B = np.outer(v_max, v_max)
    for alpha in alpha_range:
        for beta in beta_range:
            Pi = I + alpha * A - beta * B
            if np.linalg.eigvalsh(Pi).min() <= 0:
                continue
            s = full_matrix_spread(H, Pi)
            if s < best_spread:
                best_spread = s
                best_ab = (float(alpha), float(beta))
    return best_ab, best_spread


# ---------- driver ----------

def run_seed(seed: int, verbose: bool = True) -> dict[str, list[float]]:
    X = make_patterns(K=16, N=64, seed=seed)
    R = build_default_R(N=64, seed=seed)
    model = PCAMModel(X, R)

    results = {
        "base": [],
        "diag_best": [],
        "boost_bot": [],
        "suppress_top": [],
        "double": [],
    }
    rows: list[dict] = []

    for idx in range(len(X)):
        eq = model.find_equilibrium(X[idx])
        H = model.hessian(eq)
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        v_min = eigvecs[:, 0]
        v_max = eigvecs[:, -1]

        # Baseline: Π = I.
        s_base = full_matrix_spread(H, np.eye(model.N))

        # Best diagonal π — try diag(H⁻¹) and a top/bot eigvec lift, take the best.
        eigvals_c = np.clip(eigvals, 1e-9, None)
        H_inv = eigvecs @ np.diag(1.0 / eigvals_c) @ eigvecs.T
        pi_diag = model.clip_and_normalise(np.diag(H_inv))
        pi_lift = model.clip_and_normalise(np.exp(4.0 * v_min ** 2 - 2.0 * v_max ** 2))
        s_d1 = _symmetrised_spread(pi_diag, H) or float("inf")
        s_d2 = _symmetrised_spread(pi_lift, H) or float("inf")
        s_diag = min(s_d1, s_d2)

        # Rank-1 strategies.
        alpha_b, s_boost = optimize_rank1(H, v_min, mode="boost",
                                          alpha_range=ALPHA_BOOST)
        alpha_s, s_suppress = optimize_rank1(H, v_max, mode="suppress",
                                             alpha_range=ALPHA_SUPPRESS)
        (a, b), s_double = optimize_double_rank1(H, v_min, v_max)

        results["base"].append(s_base)
        results["diag_best"].append(s_diag)
        results["boost_bot"].append(s_boost)
        results["suppress_top"].append(s_suppress)
        results["double"].append(s_double)

        rows.append({
            "seed": seed,
            "pattern": idx,
            "base": s_base,
            "diag_best": s_diag,
            "boost_bot_alpha": alpha_b,
            "boost_bot_spread": s_boost,
            "suppress_top_alpha": alpha_s,
            "suppress_top_spread": s_suppress,
            "double_alpha": a,
            "double_beta": b,
            "double_spread": s_double,
        })

        if verbose and idx < 3:
            print(f"  pat {idx}: base={s_base:.2f}")
            print(f"    diag_best={s_diag:.2f} ({s_base / s_diag:.2f}×)")
            print(f"    boost_bot(α={alpha_b:.2f})={s_boost:.2f} "
                  f"({s_base / s_boost:.2f}×)")
            print(f"    suppress_top(α={alpha_s:.2f})={s_suppress:.2f} "
                  f"({s_base / s_suppress:.2f}×)")
            print(f"    double(α={a:.2f}, β={b:.2f})={s_double:.2f} "
                  f"({s_base / s_double:.2f}×)")

    if verbose:
        mb = float(np.mean(results["base"]))
        print(f"\n  MEAN reductions (seed {seed}):")
        for key in ("diag_best", "boost_bot", "suppress_top", "double"):
            m = float(np.mean(results[key]))
            print(f"    {key:15s}: {mb / m:.3f}×  (spread {m:.2f})")
        print()

    results["_rows"] = rows  # type: ignore[assignment]
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rank-1 precision diagnostic.")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--csv-out",
                    default=os.path.join(ROOT, "proofs", "diag_rank1.csv"))
    args = ap.parse_args(argv)

    print()
    print("Rank-1 precision diagnostic — diagonal vs. full-matrix Π")
    print(f"  seeds: {args.seeds}")
    print()

    all_rows: list[dict] = []
    aggregate = {k: [] for k in ("base", "diag_best", "boost_bot",
                                  "suppress_top", "double")}

    for seed in args.seeds:
        print(f"── seed {seed} ──")
        res = run_seed(seed, verbose=True)
        all_rows.extend(res.pop("_rows"))  # type: ignore[arg-type]
        for k, v in res.items():
            aggregate[k].extend(v)

    # CSV.
    fieldnames = ["seed", "pattern", "base", "diag_best",
                  "boost_bot_alpha", "boost_bot_spread",
                  "suppress_top_alpha", "suppress_top_spread",
                  "double_alpha", "double_beta", "double_spread"]
    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    with open(args.csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"  wrote {os.path.relpath(args.csv_out, ROOT)}")
    print()

    # Aggregate summary.
    mb = float(np.mean(aggregate["base"]))
    print("=" * 60)
    print("AGGREGATE across all seeds × patterns:")
    for key in ("diag_best", "boost_bot", "suppress_top", "double"):
        m = float(np.mean(aggregate[key]))
        print(f"  {key:15s}: {mb / m:.3f}×  (spread {m:.2f})")
    print()
    print("CONCLUSION:")
    print("  If boost_bot or suppress_top >> diag_best, the diagonal")
    print("  constraint is THE bottleneck and full-matrix Π would help.")
    print("  If they sit close together, the H structure is fundamentally")
    print("  ill-conditioned and no linear preconditioner can fix it.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
