import numpy as np
from scipy.optimize import minimize
from pcam_model import PCAMModel, build_default_R
from data import make_patterns
from checks import per_pattern_spread

def empirical_anisotropy(log_pi, model, patterns, seed=42):
    pi = np.exp(log_pi)
    pi = np.clip(pi, 0.1, 10.0)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(patterns), size=min(16, len(patterns)), replace=False)
    spreads = []
    for idx in indices:
        s = per_pattern_spread(model, pi, patterns[idx])
        if s is not None:
            spreads.append(s)
    return float(np.mean(spreads)) if spreads else 999.0

def optimize_pi(model, patterns, seed=42):
    init_log_pi = np.zeros(patterns.shape[1])

    def objective(log_pi):
        spread = empirical_anisotropy(log_pi, model, patterns, seed)
        penalty = 1e-5 * np.sum(log_pi**2)
        return spread + penalty

    print("Optimizing Pi...")
    result = minimize(
        objective,
        init_log_pi,
        method='L-BFGS-B',
        options={'maxiter': 150, 'ftol': 1e-9}
    )
    optimal_pi = np.exp(result.x)
    optimal_pi = np.clip(optimal_pi / optimal_pi.mean(), 0.1, 10.0)
    return optimal_pi, result.fun

for seed in [42, 101, 7]:
    K, N = 16, 64
    X = make_patterns(K=K, N=N, seed=seed)
    R = build_default_R(N=N, seed=seed)
    model = PCAMModel(X, R)

    baseline = empirical_anisotropy(np.zeros(N), model, X, seed)
    pi_opt, fun = optimize_pi(model, X, seed)
    agent = empirical_anisotropy(np.log(pi_opt), model, X, seed)

    print(f"seed={seed}: baseline={baseline:.4f}, agent={agent:.4f}, ratio={baseline/agent:.4f}x")
    print(f"  pi range: [{pi_opt.min():.4f}, {pi_opt.max():.4f}]")
