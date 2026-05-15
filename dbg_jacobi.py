"""Verify Jacobi preconditioner reduces the bench spread metric."""
import numpy as np
from pcam_model import PCAMModel, build_default_R
from data import make_patterns


def measure_spread(H, pi, model):
    pi = model.clip_and_normalise(pi)
    S = (np.sqrt(pi)[:, None] * H) * np.sqrt(pi)[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return float(eigs.max() / eigs.min())


for seed in [42, 101, 7, 13, 31]:
    K, N = 16, 64
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)

    print(f"\n=== seed {seed} ===")
    rng = np.random.default_rng(seed)
    spreads_id, spreads_jac, spreads_jac_clip = [], [], []
    for k in range(K):
        # Probe-style perturbation matches anisotropy_spread()
        probe = X[k] + rng.standard_normal(N) * 0.05
        probe = probe / np.linalg.norm(probe)
        H = model.hessian(X[k])
        H = 0.5 * (H + H.T)

        # baseline (Π = I)
        s_id = measure_spread(H, np.ones(N), model)
        spreads_id.append(s_id)

        # Jacobi raw 1/diag(H)
        diagH = np.diag(H)
        diagH_safe = np.where(np.abs(diagH) < 1e-6, 1.0, diagH)
        pi_jac = 1.0 / diagH_safe
        s_j = measure_spread(H, pi_jac, model)
        spreads_jac.append(s_j)

        # Jacobi with absolute value (handles indefinite H)
        pi_jac_abs = 1.0 / (np.abs(diagH) + 1e-3)
        s_jc = measure_spread(H, pi_jac_abs, model)
        spreads_jac_clip.append(s_jc)

    base = np.mean(spreads_id)
    jac = np.mean(spreads_jac)
    jac_abs = np.mean(spreads_jac_clip)
    print(f"  baseline (Π=I) mean spread:     {base:8.3f}")
    print(f"  Jacobi 1/diag(H) mean spread:   {jac:8.3f}  reduction = {base/jac:.2f}x")
    print(f"  Jacobi 1/|diag(H)|+ε mean:      {jac_abs:8.3f}  reduction = {base/jac_abs:.2f}x")
