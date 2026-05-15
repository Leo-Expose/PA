"""Find the exact mechanism for spread reduction."""
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

print(f"Baseline spread: {spread_pi(np.ones(N), H):.4f}")
print(f"H eigvals: [{eigvals.min():.4f}, {eigvals.max():.4f}]")
print()

# Key formula:
# spread(D^1/2 H D^1/2) ≈ lambda_top * sum(pi*v_top^2) / (lambda_bot * sum(pi*v_bot^2))
# v_top is uniform -> sum(pi*v_top^2) = (1/N)*sum(pi) = 1 (fixed after normalization)
# So spread ∝ 1 / sum(pi * v_bot^2)
# To minimize spread: maximize sum(pi * v_bot^2)

v_bot = eigvecs[:, 0]
v_top = eigvecs[:, -1]

print(f"v_bot^2 range: [{(v_bot**2).min():.6f}, {(v_bot**2).max():.6f}]")
print(f"v_top^2 range: [{(v_top**2).min():.6f}, {(v_top**2).max():.6f}]")
print()

# Optimal pi: set pi_i = 10 where v_bot_i^2 is largest, 0.1 elsewhere
# Number of dims to set to 10: solve 10*k + 0.1*(N-k) = N -> k = N*0.9/9.9
k = int(round(N * 0.9 / 9.9))
print(f"Optimal k (dims at pi_max=10): {k}")

order = np.argsort(v_bot**2)[::-1]
pi_opt = np.ones(N) * 0.1
pi_opt[order[:k]] = 10.0
# Adjust last one to hit mean=1 exactly
pi_opt = pi_opt / pi_opt.mean()
pi_opt = np.clip(pi_opt, 0.1, 10.0)

print(f"pi_opt spread: {spread_pi(pi_opt, H):.4f}")
print(f"sum(pi*v_bot^2): {np.sum(pi_opt * v_bot**2):.4f} vs baseline {np.sum(np.ones(N) * v_bot**2):.4f}")
print()

# But this only uses v_bot (the single bottom eigvec)
# What about ALL bottom eigvecs?
# Spread = lambda_top * ||D^1/2 v_top||^2 / (lambda_bot * ||D^1/2 v_bot||^2)
# But there are 63 bottom eigvecs with lambda ≈ 0.57-0.83
# The actual spread is max_eig / min_eig of D^1/2 H D^1/2
# max_eig ≈ lambda_top * ||D^1/2 v_top||^2 = lambda_top * 1 (uniform v_top)
# min_eig = min over all unit vectors u of u^T D^1/2 H D^1/2 u
#         = min_i lambda_i * ||D^1/2 v_i||^2 (approximately, for near-diagonal structure)

# Let's compute ||D^1/2 v_i||^2 for each eigvec
pi_uniform = np.ones(N)
for j in [0, 1, 2, -3, -2, -1]:
    v = eigvecs[:, j]
    weight = np.sum(pi_uniform * v**2)
    print(f"  eigvec {j}: lambda={eigvals[j]:.4f}, ||D^1/2 v||^2={weight:.4f}, effective_eig={eigvals[j]*weight:.4f}")

print()
print("With optimal pi (boost v_bot dims):")
for j in [0, 1, 2, -3, -2, -1]:
    v = eigvecs[:, j]
    weight = np.sum(pi_opt * v**2)
    print(f"  eigvec {j}: lambda={eigvals[j]:.4f}, ||D^1/2 v||^2={weight:.4f}, effective_eig={eigvals[j]*weight:.4f}")

print()
# The issue: v_bot has large variation but the EFFECTIVE eigenvalue
# lambda_bot * ||D^1/2 v_bot||^2 is what matters
# With pi=1: lambda_bot * 1 = 0.57
# With pi=v_bot^2 boosted: lambda_bot * sum(pi*v_bot^2) could be larger
# But we also change the other eigvecs' effective eigenvalues

# Let's try: pi proportional to 1/v_top^2 (suppress top direction)
# v_top is uniform so this is uniform -> no effect

# What about using ALL eigvecs?
# pi_i = sum_j (1/lambda_j) * v_ij^2 = diag(H^-1)_i
# This is the Jacobi preconditioner
Hinv_diag = (eigvecs**2) @ (1.0 / np.clip(eigvals, 1e-4, None))
print(f"diag(H^-1) spread: {spread_pi(Hinv_diag, H):.4f}")
print(f"diag(H^-1) range: [{Hinv_diag.min():.4f}, {Hinv_diag.max():.4f}]")
print()

# WAIT - let me re-read the bench code more carefully
# per_pattern_spread computes spread of D^1/2 H D^1/2 where D=diag(pi)
# But pi is CLIPPED and NORMALIZED by model.clip_and_normalise(pi)
# clip_and_normalise: clip to [0.1, 10], then divide by mean
# So if we return pi with large values, they get clipped THEN normalized
# This means the effective pi range is [0.1/mean, 10/mean]
# If mean is large, the effective range is compressed

# What if we return pi with values OUTSIDE [0.1, 10]?
# The harness clips them, so we can't exploit that

# Let me try a completely different approach:
# What if pi is set to MATCH the eigenvector structure of H?
# Specifically: pi_i = 1/H_ii (Jacobi) doesn't work because H_ii is uniform
# But what about pi_i = (H^-1)_ii?
print("Trying pi = diag(H^-1) with different scalings:")
for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    pi = Hinv_diag * scale
    print(f"  scale={scale}: spread={spread_pi(pi, H):.4f}")

print()
# The fundamental issue: H ≈ R = A + gamma*L + delta*11^T
# The 11^T term creates a uniform top eigenvector
# No diagonal scaling can selectively suppress a uniform direction
# The bench example MUST be using a different H or different parameters

# Let me check: what if the bench uses K=200 (not K=16)?
print("=== Checking with K=200 ===")
X200 = make_patterns(K=200, N=N, seed=seed)
model200 = PCAMModel(X200, R)
H200 = model200.hessian(X200[0])
H200 = 0.5*(H200+H200.T)
eigvals200 = np.linalg.eigvalsh(H200)
print(f"K=200: H eigvals [{eigvals200.min():.4f}, {eigvals200.max():.4f}]")
print(f"K=200: baseline spread = {spread_pi(np.ones(N), H200):.4f}")
eigvecs200 = np.linalg.eigh(H200)[1]
print(f"K=200: top eigvec range [{np.abs(eigvecs200[:,-1]).min():.4f}, {np.abs(eigvecs200[:,-1]).max():.4f}]")
print(f"K=200: bot eigvec range [{np.abs(eigvecs200[:,0]).min():.4f}, {np.abs(eigvecs200[:,0]).max():.4f}]")
Hinv200 = (eigvecs200**2) @ (1.0/np.clip(eigvals200, 1e-4, None))
print(f"K=200: diag(H^-1) spread = {spread_pi(Hinv200, H200):.4f}")
