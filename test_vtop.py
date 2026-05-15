import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns

seed = 42
K, N = 16, 64
X = make_patterns(K=K, N=N, seed=seed)
R = build_default_R(N=N, seed=seed)
model = PCAMModel(X, R)

H = model.hessian(X[0])
H = 0.5 * (H + H.T)
eigvals, eigvecs = np.linalg.eigh(H)
v_top = eigvecs[:, -1]
print("v_top norm:", np.linalg.norm(v_top))
print("X[0] norm:", np.linalg.norm(X[0]))
print("cosine(v_top, X[0]):", np.abs(np.dot(v_top, X[0])))

# Check components of X[0] vs v_top
print("v_top absolute components min/max:", np.abs(v_top).min(), np.abs(v_top).max())
print("X[0] absolute components min/max:", np.abs(X[0]).min(), np.abs(X[0]).max())
