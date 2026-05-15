"""Prove why diagonal pi cannot reduce spread on this H, and find the real mechanism."""
import numpy as np
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

print("=== Hessian structure ===")
print(f"eigvals: {eigvals[[0,1,2,-3,-2,-1]]}")
c = eigvals[:-1].mean()
b = eigvals[-1] - c
print(f"H ≈ c*I + b*v_top*v_top^T: c={c:.4f}, b={b:.4f}")
print(f"v_top range: [{np.abs(eigvecs[:,-1]).min():.4f}, {np.abs(eigvecs[:,-1]).max():.4f}]")
print(f"Theoretical spread floor = (c+b)/c = {(c+b)/c:.4f}")
print(f"Actual baseline spread = {spread_pi(np.ones(N), H):.4f}")
print()

# The bench's anisotropy_spread() uses a PROBE query to get pi,
# then measures spread at the STORED PATTERN.
# But what if the bench example uses a DIFFERENT Hessian evaluation point?
# Let's check: what if we evaluate H at a MIDPOINT between two patterns?
print("=== What if H is evaluated at a non-attractor point? ===")
# At a midpoint between pattern 0 and its twin (pattern K//2)
mid = (X[0] + X[K//2]) / 2
mid = mid / np.linalg.norm(mid)
H_mid = model.hessian(mid)
H_mid = 0.5 * (H_mid + H_mid.T)
eigvals_mid = np.linalg.eigvalsh(H_mid)
print(f"H at midpoint eigvals: [{eigvals_mid.min():.4f}, {eigvals_mid.max():.4f}]")
print(f"H at midpoint spread (pi=I): {spread_pi(np.ones(N), H_mid):.4f}")

# Check eigvec structure at midpoint
eigvecs_mid = np.linalg.eigh(H_mid)[1]
print(f"H_mid top eigvec range: [{np.abs(eigvecs_mid[:,-1]).min():.4f}, {np.abs(eigvecs_mid[:,-1]).max():.4f}]")
print(f"H_mid bottom eigvec range: [{np.abs(eigvecs_mid[:,0]).min():.4f}, {np.abs(eigvecs_mid[:,0]).max():.4f}]")

# Try pi = diag(H_mid^-1) on H_mid
eigvals_m, eigvecs_m = np.linalg.eigh(H_mid)
Hinv_diag_mid = (eigvecs_m**2) @ (1.0 / np.clip(eigvals_m, 1e-4, None))
print(f"pi=diag(H_mid^-1) spread on H_mid: {spread_pi(Hinv_diag_mid, H_mid):.4f}")
print(f"pi=diag(H_mid^-1) spread on H[0]: {spread_pi(Hinv_diag_mid, H):.4f}")
print()

# What if the bench evaluates H at the QUERY (corrupted), not the pattern?
print("=== H at corrupted query ===")
rng = np.random.default_rng(seed)
from data import corrupt
q = corrupt(X[0], 0.7, rng)
H_q = model.hessian(q)
H_q = 0.5 * (H_q + H_q.T)
eigvals_q, eigvecs_q = np.linalg.eigh(H_q)
print(f"H at corrupted query eigvals: [{eigvals_q.min():.4f}, {eigvals_q.max():.4f}]")
print(f"H at query spread (pi=I): {spread_pi(np.ones(N), H_q):.4f}")
print(f"H_q top eigvec range: [{np.abs(eigvecs_q[:,-1]).min():.4f}, {np.abs(eigvecs_q[:,-1]).max():.4f}]")

# pi = diag(H_q^-1) on H[0]
Hinv_diag_q = (eigvecs_q**2) @ (1.0 / np.clip(eigvals_q, 1e-4, None))
print(f"pi=diag(H_q^-1) spread on H[0]: {spread_pi(Hinv_diag_q, H):.4f}")
print()

# The bench's anisotropy check: probe = pattern + 0.05*noise, normalized
# Then pi = agent.predict_precision(probe)
# Then spread = per_pattern_spread(model, pi, pattern)
# per_pattern_spread evaluates H at 'pattern' (the stored pattern)
# So the H in the spread metric is ALWAYS H(stored_pattern)
# This means we need pi that reduces spread of D^1/2 H(X[k]) D^1/2
# And we've proven this is impossible when H(X[k]) ≈ c*I + b*uniform*uniform^T

# CONCLUSION: The bench example output (8.42x) must use a DIFFERENT H structure
# Let's check if maybe the bench uses H at the EQUILIBRIUM after running dynamics
print("=== H at equilibrium after running dynamics ===")
pi_test = np.ones(N)
a_eq = model.run(q, pi_test, u_const=q)
H_eq = model.hessian(a_eq)
H_eq = 0.5 * (H_eq + H_eq.T)
eigvals_eq, eigvecs_eq = np.linalg.eigh(H_eq)
print(f"H at equilibrium eigvals: [{eigvals_eq.min():.4f}, {eigvals_eq.max():.4f}]")
print(f"H at equilibrium spread (pi=I): {spread_pi(np.ones(N), H_eq):.4f}")
print(f"H_eq top eigvec range: [{np.abs(eigvecs_eq[:,-1]).min():.4f}, {np.abs(eigvecs_eq[:,-1]).max():.4f}]")
Hinv_diag_eq = (eigvecs_eq**2) @ (1.0 / np.clip(eigvals_eq, 1e-4, None))
print(f"pi=diag(H_eq^-1) spread on H_eq: {spread_pi(Hinv_diag_eq, H_eq):.4f}")
print(f"pi=diag(H_eq^-1) spread on H[0]: {spread_pi(Hinv_diag_eq, H):.4f}")
