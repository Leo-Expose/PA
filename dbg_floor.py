"""
The theoretical floor for H = c*I + b*v*v^T with v uniform is (c+b)/c.
Baseline spread is 12.15 but floor is 9.84. So there IS room.
The question: what pi achieves the floor?

For H = c*I + b*v*v^T:
  D^1/2 H D^1/2 = c*D + b*(D^1/2 v)(D^1/2 v)^T
  
Eigenvalues of this matrix:
  - N-1 eigenvalues = c*pi_i for i not in span of D^1/2 v
  - 1 eigenvalue = c*pi_j + b*||D^1/2 v||^2 for the direction of D^1/2 v

After mean-normalization: sum(pi) = N, so ||D^1/2 v||^2 = (1/N)*sum(pi) = 1
(since v is uniform: v_i = 1/sqrt(N))

So the outlier eigenvalue = c*pi_j + b (where pi_j is the component in v direction)
But v is uniform, so the outlier is a mix of all pi_i.

Actually for rank-1 perturbation:
  eigs of (c*D + b*u*u^T) where u = D^1/2 * v:
  - N-1 smallest: c*pi_i (the N-1 smallest pi_i values)
  - 1 largest: c*pi_max + b*||u||^2 = c*pi_max + b (approximately, for uniform v)

Wait, that's not right either. Let me compute exactly.
"""
import numpy as np
from scipy.optimize import minimize
from pcam_model import PCAMModel, build_default_R
from data import make_patterns


def spread_pi(pi, H):
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / (pi.mean() + 1e-12)
    pi = np.clip(pi, 0.1, 10.0)
    sq = np.sqrt(pi)
    S = (sq[:, None] * H) * sq[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return float(eigs.max() / eigs.min())


seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)
H = model.hessian(X[0])
H = 0.5 * (H + H.T)
eigvals, eigvecs = np.linalg.eigh(H)

print("=== Exact rank-1 analysis ===")
# H = sum_j lambda_j * v_j * v_j^T
# D^1/2 H D^1/2 = sum_j lambda_j * (D^1/2 v_j)(D^1/2 v_j)^T
# For uniform v_top: (D^1/2 v_top)_i = sqrt(pi_i) * (1/sqrt(N))
# ||D^1/2 v_top||^2 = sum(pi_i)/N = 1 (after normalization)
# So the outlier eigenvalue is ALWAYS lambda_top * 1 = lambda_top (approx)
# regardless of pi! That's why we can't reduce it.

# But wait - the bottom eigenvalues are NOT uniform eigvecs
# v_bot has large variation: [0.0026, 0.2638]
# So (D^1/2 v_bot)_i = sqrt(pi_i) * v_bot_i
# ||D^1/2 v_bot||^2 = sum(pi_i * v_bot_i^2)
# This CAN be changed by pi!

# The spread is lambda_top / lambda_bot_effective
# lambda_bot_effective = lambda_bot * ||D^1/2 v_bot||^2 (approximately)
# To maximize lambda_bot_effective: maximize sum(pi_i * v_bot_i^2)
# Subject to mean(pi) = 1, 0.1 <= pi_i <= 10

v_bot = eigvecs[:, 0]
v_top = eigvecs[:, -1]
print(f"v_bot^2 range: [{(v_bot**2).min():.6f}, {(v_bot**2).max():.6f}]")
print(f"v_top^2 range: [{(v_top**2).min():.6f}, {(v_top**2).max():.6f}]")
print()

# Strategy: maximize sum(pi_i * v_bot_i^2) = boost where v_bot is large
# This lifts the bottom eigenvalue, reducing spread
pi_boost_bot = v_bot**2 / (v_bot**2).mean()
print(f"pi = v_bot^2 (boost bottom): spread = {spread_pi(pi_boost_bot, H):.4f}")

# Strategy: suppress where v_top is large (but v_top is uniform, so this does nothing)
pi_suppress_top = 1.0 / (v_top**2 + 1e-3)
print(f"pi = 1/v_top^2 (suppress top): spread = {spread_pi(pi_suppress_top, H):.4f}")

# Combined: boost bottom AND suppress top
pi_combined = v_bot**2 / (v_top**2 + 1e-3)
print(f"pi = v_bot^2 / v_top^2: spread = {spread_pi(pi_combined, H):.4f}")

# What about boosting ALL bottom eigenvectors?
# pi_i = sum_j (1/lambda_j) * v_ij^2  -- this is diag(H^-1)
Hinv_diag = (eigvecs**2) @ (1.0 / np.clip(eigvals, 1e-4, None))
print(f"pi = diag(H^-1): spread = {spread_pi(Hinv_diag, H):.4f}")

# What about pi_i = 1/lambda_bot * v_bot_i^2 only?
pi_bot_only = (1.0/eigvals[0]) * v_bot**2
print(f"pi = (1/lambda_bot)*v_bot^2: spread = {spread_pi(pi_bot_only, H):.4f}")

# Gradient-based: minimize spread w.r.t. log(pi)
def obj(log_pi):
    return spread_pi(np.exp(log_pi), H)

best = float('inf')
best_pi = None
rng = np.random.default_rng(42)
for trial in range(30):
    # Initialize toward v_bot^2
    log0 = np.log(np.clip(v_bot**2 + 1e-3, 0.1, 10.0)) + rng.standard_normal(N) * 0.5
    res = minimize(obj, log0, method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6})
    if res.fun < best:
        best = res.fun
        best_pi = np.exp(res.x)
        print(f"  trial {trial}: spread = {res.fun:.4f}")

print(f"\nBest found: {best:.4f} (baseline: {spread_pi(np.ones(N), H):.4f})")
print(f"Reduction: {spread_pi(np.ones(N), H)/best:.3f}x")
print(f"Theoretical floor: {eigvals[-1]/eigvals[0]:.4f}")
