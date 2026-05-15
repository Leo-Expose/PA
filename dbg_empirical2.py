import numpy as np
from scipy.optimize import minimize
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from checks import per_pattern_spread

def run_dynamics_with_pi(model, query, pi, steps=30):
    a = query.copy()
    for _ in range(steps):
        g = model.gradient(a)
        a = a - model.dt * (pi * g)
        a = np.clip(a, -1+1e-6, 1-1e-6)
    return a

def covariance_spread(pi, model, queries):
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    attractors = np.array([run_dynamics_with_pi(model, q, pi) for q in queries])
    cov = np.cov(attractors.T)
    eigvals = np.maximum(np.linalg.eigvalsh(cov), 1e-12)
    return float(eigvals[-1] / eigvals[0])

for seed in [42, 101]:
    K, N = 16, 64
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)

    rng = np.random.default_rng(seed)
    queries = []
    for k in range(K):
        for noise in [0.5, 0.7, 0.8]:
            q = X[k].copy()
            mask = rng.random(N) < noise
            q[mask] = rng.standard_normal(mask.sum()) * 0.5
            q = q / (np.linalg.norm(q) + 1e-9)
            queries.append(q)
    queries = np.array(queries)

    base_cov = covariance_spread(np.ones(N), model, queries)
    base_pps = np.mean([per_pattern_spread(model, np.ones(N), X[k]) for k in range(K)])
    print(f"seed={seed}: baseline cov_spread={base_cov:.4f}, per_pattern_spread={base_pps:.4f}")

    def objective(log_pi):
        pi = np.exp(log_pi)
        return covariance_spread(pi, model, queries) + 1e-4 * np.sum(log_pi**2)

    result = minimize(objective, np.zeros(N), method='L-BFGS-B',
                      options={'maxiter': 100, 'ftol': 1e-6})
    pi_opt = np.exp(result.x)
    pi_opt = np.clip(pi_opt / pi_opt.mean(), 0.1, 10.0)

    agent_cov = covariance_spread(pi_opt, model, queries)
    agent_pps = np.mean([per_pattern_spread(model, pi_opt, X[k]) for k in range(K)])

    print(f"seed={seed}: agent   cov_spread={agent_cov:.4f} (ratio={base_cov/agent_cov:.4f}x)")
    print(f"seed={seed}: agent   per_pattern_spread={agent_pps:.4f} (ratio={base_pps/agent_pps:.4f}x)")
    print(f"  pi range: [{pi_opt.min():.4f}, {pi_opt.max():.4f}], std={pi_opt.std():.4f}")
    print()
