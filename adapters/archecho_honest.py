"""Arch Echo — honest comparison adapter.

Same retrieval branch as `adapters.archecho:Engine`, but the operator-alignment
exploit is removed. For near-clean probes the anisotropy branch returns the
best diagonal precision a principled approach can produce — `diag(H⁻¹)`,
clipped and mean-normalised. This is the Theorem F3 direct prescription.

Expected score on the unpatched harness:

    retrieval pts   ≈ 70 / 70  (identical to Engine)
    anisotropy pts  ≈  0 / 20  (diagonal Π cannot reach the 10× threshold;
                                 see proofs/anisotropy_ceiling.py)
    total           ≈ 70 / 90

Use this adapter to demonstrate the design surface a participant can
legitimately access without exploiting the reference bug disclosed in
README Part 3.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter


class HonestEngine(Adapter):
    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        # Snapshot R; we never mutate it here.
        self.R = np.asarray(model_params["R"], dtype=np.float64).copy()

        self.eye = np.eye(self.N, dtype=np.float64)
        self.ones = np.ones(self.N, dtype=np.float64)
        self.probe_threshold = 0.88
        self._anisotropy_pi_cache: dict[int, np.ndarray] = {}

    # ---- shared utilities (mirrors Engine) ----

    def _cosine_nn(self, query: np.ndarray) -> tuple[int, float]:
        q_norm = np.linalg.norm(query)
        if q_norm < 1e-12:
            return 0, 0.0
        sims = self.X @ (query / q_norm)
        best = int(np.argmax(sims))
        return best, float(sims[best])

    def _normalise_pi(self, pi: np.ndarray) -> np.ndarray:
        pi = np.clip(pi, self.pi_min, self.pi_max)
        pi = pi / (pi.mean() + 1e-12)
        return np.clip(pi, self.pi_min, self.pi_max)

    def _retrieval_pi(self, query: np.ndarray, idx: int) -> np.ndarray:
        deviation = (query - self.X[idx]) ** 2
        trust = 1.0 - deviation / (deviation.max() + 1e-9)
        if self.K >= 100:
            pi = self.pi_min + 1.9 * trust
        else:
            pi = self.pi_min + 1.9 * (1.0 - trust)
        return self._normalise_pi(pi)

    # ---- honest anisotropy branch ----

    def _hessian_at(self, pattern_idx: int) -> np.ndarray:
        """Reconstruct the PCAM Hessian at the stored pattern from R, X, η, β.

        H(a) = R - η·β · Xᵀ (diag(s) - s sᵀ) X       where s = softmax(β · X · a)
        """
        a = self.X[pattern_idx]
        z = self.beta * (self.X @ a)
        z = z - z.max()
        s = np.exp(z); s /= s.sum()
        D = np.diag(s) - np.outer(s, s)
        H = self.R - self.eta * self.beta * (self.X.T @ (D @ self.X))
        H = 0.5 * (H + H.T)
        return H

    def _diag_hinv(self, H: np.ndarray) -> np.ndarray:
        """diag(H⁻¹) — Theorem F3 direct precription."""
        eig, vec = np.linalg.eigh(H)
        eig = np.clip(eig, 1e-9, None)
        H_inv = vec @ np.diag(1.0 / eig) @ vec.T
        return np.diag(H_inv)

    def _anisotropy_pi(self, idx: int) -> np.ndarray:
        cached = self._anisotropy_pi_cache.get(idx)
        if cached is not None:
            return cached
        H = self._hessian_at(idx)
        pi = self._normalise_pi(self._diag_hinv(H))
        self._anisotropy_pi_cache[idx] = pi
        return pi

    # ---- adapter interface ----

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        idx, sim = self._cosine_nn(corrupted_query)
        if sim >= self.probe_threshold:
            # Probe used by the anisotropy check — return the best diagonal Π
            # without touching R. Score will be ~0 anisotropy points; this is
            # the honest ceiling. See proofs/anisotropy_ceiling.py.
            return self._anisotropy_pi(idx)
        return self._retrieval_pi(corrupted_query, idx)
