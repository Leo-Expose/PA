"""Brute-force find optimal diagonal preconditioner via random search + local refine.
This tells us the upper bound of what is achievable per pattern."""
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns


def spread(H, pi):
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    S = (np.sqrt(pi)[:, None] * H) * np.sqrt(pi)[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return float(eigs.max() / eigs.min())


def coord_descent(H, pi0, iters=200, step=0.1):
    pi = pi0.copy()
    best = spread(H, pi)
    N = len(pi)
    for it in range(iters):
        improved = False
        for i in range(N):
            for delta in [step, -step, step * 0.3, -step * 0.3]:
                pi_try = pi.copy()
                pi_try[i] = max(0.1, min(10.0, pi_try[i] + delta))
                s = spread(H, pi_try)
                if s < best:
                    best = s
                    pi = pi_try
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-3:
                break
    return pi, best


seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)

# Take a single pattern's Hessian
H = model.hessian(X[0])
H = 0.5 * (H + H.T)

print(f"baseline spread (Π=I): {spread(H, np.ones(N)):.4f}")

# Random search
rng = np.random.default_rng(0)
best_overall = float("inf")
best_pi = None
for trial in range(50):
    pi0 = np.exp(rng.standard_normal(N) * 1.0)
    pi_opt, s = coord_descent(H, pi0, iters=50)
    if s < best_overall:
        best_overall = s
        best_pi = pi_opt
        print(f"  trial {trial}: spread = {s:.4f}")

print(f"\nbest spread found: {best_overall:.4f}")
print(f"reduction = {spread(H, np.ones(N)) / best_overall:.2f}x")
print(f"pi range: [{best_pi.min():.3f}, {best_pi.max():.3f}]")
print(f"pi values:", np.round(best_pi, 2))

# Compare with X[0]^2 boost
print("\n--- check pattern-coordinate strategies ---")
for power in [0.5, 1.0, 2.0]:
    pi = np.abs(X[0]) ** power + 1e-3
    print(f"  |x|^{power}: spread = {spread(H, pi):.4f}")
    pi = 1.0 / (np.abs(X[0]) ** power + 1e-3)
    print(f"  |x|^-{power}: spread = {spread(H, pi):.4f}")
