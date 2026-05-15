import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)

H_pat = model.hessian(X[0])
H_pat = 0.5 * (H_pat + H_pat.T)

# Create a random mask
rng = np.random.default_rng(seed)
mask = rng.random(N) > 0.5

# Augmented Hessian
H_full = H_pat + 400.0 * np.diag(mask)
H_full = 0.5 * (H_full + H_full.T)

eigvals, eigvecs = np.linalg.eigh(H_full)
v_top = eigvecs[:, -1]
print("v_top (H_full) absolute components min/max:", np.abs(v_top).min(), np.abs(v_top).max())
