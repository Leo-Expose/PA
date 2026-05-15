from adapter import Adapter
import numpy as np
import time

MASK_THRESHOLD = 1e-6
HESSIAN_EPS    = 1e-4

class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        t0 = time.time()

        self.X     = stored_patterns           # (K, 64)
        self.K, self.N = stored_patterns.shape
        self.model = model_params              # frozen PCAM model

        # Lesson 1: Verify API before expensive Hessian loop
        test_grad = self._grad(self.X[0])
        assert test_grad.shape == (self.N,), \
            f"gradient() returned {test_grad.shape}, expected ({self.N},)"

        # Pre-compute full geometry for every stored pattern
        self.pi_geo = np.stack([
            self._pi_from_hessian(
                self._full_hessian(self.X[k])
            )
            for k in range(self.K)
        ])  # (K, N)

        # Lesson 2: Init runtime warning
        elapsed = time.time() - t0
        if elapsed > 30:
            print(f"[WARN] __init__ took {elapsed:.1f}s")

    # ── Hessian via central finite differences ─────────────────────
    def _grad(self, x):
        """Gradient of the frozen PCAM energy at x."""
        return self.model.gradient(x)  # provided by starter kit

    def _full_hessian(self, x):
        N = x.shape[0]
        H = np.zeros((N, N))
        for i in range(N):
            xp, xm = x.copy(), x.copy()
            xp[i] += HESSIAN_EPS
            xm[i] -= HESSIAN_EPS
            H[:, i] = (self._grad(xp) - self._grad(xm)) / (2 * HESSIAN_EPS)
        return (H + H.T) / 2   # enforce symmetry

    def _pi_from_hessian(self, H):
        """
        Theorem F3: precision = V diag(1/λ) Vᵀ evaluated on its diagonal.
        Balances all 64 convergence rates simultaneously.
        """
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.clip(eigvals, 1e-4, None)
        pi_eig  = 1.0 / eigvals
        pi_eig  = pi_eig / pi_eig.mean()

        # Correct diagonal extraction: diag(V * diag(pi_eig) * V^T) = (V^2) @ pi_eig
        pi_diag = (eigvecs**2) @ pi_eig

        return np.clip(pi_diag, 0.1, 10.0)

    # ── Mask-aware cosine similarity ───────────────────────────────
    def _masked_cosine(self, query):
        visible = np.abs(query) >= MASK_THRESHOLD
        if visible.sum() < 2:
            # Fallback: insufficient visible dims -> L2 on visible only, low confidence
            dists = np.linalg.norm(self.X[:, visible] - query[visible], axis=1)
            return int(np.argmin(dists)), 0.0

        X_vis = self.X[:, visible]
        q_vis = query[visible]
        norms = np.linalg.norm(X_vis, axis=1) * np.linalg.norm(q_vis) + 1e-9
        sims  = (X_vis @ q_vis) / norms
        best  = int(np.argmax(sims))
        return best, float(sims[best])

    # ── Main inference ────────────────────────────────────────────
    def predict_precision(self, corrupted_query):
        # Q-1: Mask detection
        is_masked = np.abs(corrupted_query) < MASK_THRESHOLD

        # Q-2: Mask-aware NN lookup
        best_idx, sim = self._masked_cosine(corrupted_query)

        # Q-3: Heuristic — per-dim deviation trust score
        deviation    = (corrupted_query - self.X[best_idx]) ** 2
        trust        = 1.0 - deviation / (deviation.max() + 1e-9)
        pi_heuristic = 0.1 + 1.9 * trust          # → [0.1, 2.0]
        pi_heuristic[is_masked] = 0.1             # force-low masked dims

        # Q-4: Adaptive β — no hardcoded params, with floor
        pi_geo = self.pi_geo[best_idx]
        beta   = 1.0 - np.clip(sim, 0.0, 1.0)
        beta   = np.clip(beta, 0.15, 1.0)        # floor: geometry always gets weight

        # Q-5: Log-space blend
        log_geo = np.log(np.clip(pi_geo,       0.1, 10.0))
        log_heu = np.log(np.clip(pi_heuristic, 0.1, 10.0))
        pi_raw  = np.exp(beta * log_geo + (1.0 - beta) * log_heu)

        # Q-6: Normalise + clip
        pi_final = pi_raw / (pi_raw.mean() + 1e-9)
        return np.clip(pi_final, 0.1, 10.0)
