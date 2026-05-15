"""Test if pattern coordinates can drive spread reduction."""
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

def measure_spread(H, pi, model):
    pi = model.clip_and_normalise(pi)
    S = (np.sqrt(pi)[:, None] * H) * np.sqrt(pi)[None, :]
    S = 0.5 * (S + S.T)
    eigs = np.linalg.eigvalsh(S)
    eigs = eigs[eigs > 1e-9]
    return eigs.max() / eigs.min()

base = measure_spread(H, np.ones(N), model)
print(f"Baseline spread: {base:.4f}")
print(f"\nPattern 0 components: min={np.abs(X[0]).min():.6f}, max={np.abs(X[0]).max():.6f}")
print(f"Pattern 0 x^2: min={np.min(X[0]**2):.6f}, max={np.max(X[0]**2):.6f}, ratio={np.max(X[0]**2)/np.min(X[0]**2):.1f}")

# Strategy: pi = 1/(x_i^2 + eps)
print("\n=== pi = 1/(x_i^2 + eps) ===")
for eps in [1e-6, 1e-4, 1e-3, 1e-2, 0.1]:
    pi = 1.0 / (X[0]**2 + eps)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = measure_spread(H, pi, model)
    print(f"  eps={eps:.0e}: pi=[{pi.min():.3f},{pi.max():.3f}], spread={s:.4f}, reduction={base/s:.2f}x")

# Strategy: pi = |x_i|^(-power)
print("\n=== pi = |x_i|^(-power) ===")
for power in [0.5, 1.0, 1.5, 2.0]:
    pi = 1.0 / (np.abs(X[0])**power + 1e-4)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = measure_spread(H, pi, model)
    print(f"  power={power}: spread={s:.4f}, reduction={base/s:.2f}x")

# Strategy: pi proportional to |x_i| (BOOST strong dims)
print("\n=== pi = |x_i|^power (boost strong) ===")
for power in [0.5, 1.0, 2.0, 4.0]:
    pi = np.abs(X[0])**power + 1e-6
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = measure_spread(H, pi, model)
    print(f"  power={power}: spread={s:.4f}, reduction={base/s:.2f}x")

# Strategy: binary - boost dims above median, suppress below
print("\n=== Binary threshold strategies ===")
median_x2 = np.median(X[0]**2)
for high, low in [(5.0, 0.2), (10.0, 0.1), (3.0, 0.3), (2.0, 0.5)]:
    pi = np.where(X[0]**2 > median_x2, high, low)
    pi = pi / pi.mean()
    pi = np.clip(pi, 0.1, 10.0)
    s = measure_spread(H, pi, model)
    print(f"  high={high},low={low}: spread={s:.4f}, reduction={base/s:.2f}x")

# Strategy: use softmax weights to identify the "winning direction"
print("\n=== Softmax-based strategies ===")
s_weights = model._softmax(X[0])
print(f"  Softmax at pattern 0: max={s_weights.max():.4f}, argmax={s_weights.argmax()}")
# The "winning" pattern contributes x_win to the Hessian
x_win = X[s_weights.argmax()]
print(f"  Winning pattern index: {s_weights.argmax()}")
print(f"  x_win^2: min={np.min(x_win**2):.6f}, max={np.max(x_win**2):.6f}")
