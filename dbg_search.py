"""Aggressive multi-start search for the *globally* optimal diagonal preconditioner."""
import numpy as np
from scipy.optimize import minimize
from pcam_model import PCAMModel, build_default_R
from data import make_patterns


def spread_from_logpi(log_pi, H):
    pi = np.exp(log_pi)
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    sq = np.sqrt(pi)
    S = (sq[:, None] * H) * sq[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return float(eigs.max() / eigs.min()) if len(eigs) >= 2 else 1e9


def try_optimize(H, log_pi0, N):
    res = minimize(
        spread_from_logpi, log_pi0, args=(H,),
        method="Powell",
        options={"xtol": 1e-5, "ftol": 1e-5, "maxiter": 5000},
    )
    return res.x, spread_from_logpi(res.x, H)


seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)
H = model.hessian(X[0])
H = 0.5 * (H + H.T)

base = spread_from_logpi(np.zeros(N), H)
print(f"baseline (Π=I) spread: {base:.4f}")

# Eigendecomp
eigvals, eigvecs = np.linalg.eigh(H)
v_top = eigvecs[:, -1]
print(f"top eigvec uniform? abs range [{np.abs(v_top).min():.4f}, {np.abs(v_top).max():.4f}]")

# Multi-start search
rng = np.random.default_rng(0)
best = base
best_pi = np.ones(N)
for trial in range(20):
    # try wider initial scales
    scale = 0.5 + trial * 0.2
    log0 = rng.standard_normal(N) * scale
    log1, s = try_optimize(H, log0, N)
    if s < best:
        best = s
        best_pi = np.exp(log1)
        print(f"  trial {trial} (scale {scale:.1f}): spread {s:.4f}")

print(f"\nBEST achievable spread on H[0]: {best:.4f}")
print(f"reduction ratio: {base / best:.2f}x")

# Theoretical lower bound: if H = c·I + b·u·u^T with u≈uniform, what's best?
# Smallest eig of D^(1/2) H D^(1/2) = c·π_min (interlacing)
# Largest eig ≥ c·π_max  (interlacing)
# Plus rank-1 lifts one eig by ≤ b·||D^(1/2)·u||^2 = b·(1/N)·sum(π)
# After mean=1 normalization: sum(π) = N, so ||D^(1/2)·u||^2 = 1
# spread ≥ (c·π_max + b) / c·π_min  for π_max=π_min=1: 13.8
print(f"\nIf this number is far from 1, the metric is genuinely unreachable")
print(f"with diagonal preconditioning on this H structure.")
