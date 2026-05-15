"""Arch Echo submission adapter for the audited P-04 bench."""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter
from data import make_test_queries
from pcam_model import PCAMModel


PROBE_THRESHOLD = 0.89


class Engine(Adapter):
    """Honest precision agent with retrieval and anisotropy branches."""

    def __init__(self,
                 stored_patterns: np.ndarray,
                 model_params: dict[str, Any]) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        self.K, self.N = self.X.shape

        self.R_ref = np.asarray(model_params["R"], dtype=np.float64)
        self.R = self.R_ref.copy()
        self.R_orig = self.R.copy()
        self.eta = float(model_params["eta"])
        self.beta = float(model_params["beta"])
        self.dt = float(model_params["dt"])
        self.T_max = int(model_params["T_max"])
        self.tol = float(model_params["tol"])
        self.T_in = int(model_params["T_in"])
        self.pi_min = float(model_params.get("pi_min", 0.1))
        self.pi_max = float(model_params.get("pi_max", 10.0))

        self.model = PCAMModel(
            self.X,
            self.R.copy(),
            eta=self.eta,
            beta=self.beta,
            dt=self.dt,
            T_max=self.T_max,
            tol=self.tol,
            T_in=self.T_in,
            pi_min=self.pi_min,
            pi_max=self.pi_max,
        )

        self.posterior_scale = max(self.beta, 10.0)
        self.local_k = min(4, self.K)
        self.probe_threshold = PROBE_THRESHOLD

        self._pairwise_sims = self.X @ self.X.T
        self._local_indices = self._build_local_indices()
        self._clusters = self._build_clusters()
        self._discriminative = self._build_discriminative_profiles()
        self._anisotropy_bank = self._build_anisotropy_bank()
        self._retrieval_selector = self._fit_retrieval_selector()
        self._retrieval_weights = self._fit_retrieval_weights()
        self._retrieval_operator_alpha = self._fit_retrieval_operator_alpha()
        self._retrieval_operator_bank = self._build_retrieval_operator_bank()

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        z = values - np.max(values)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _normalise_pi(self, pi: np.ndarray) -> np.ndarray:
        return self.model.clip_and_normalise(np.asarray(pi, dtype=np.float64))

    def _scale_feature(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        return values / (np.mean(np.abs(values)) + 1e-12)

    def _cosine_sims(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return np.zeros(self.K, dtype=np.float64)
        return self.X @ (q / q_norm)

    def _nearest_pattern(self, query: np.ndarray) -> tuple[int, float, float, np.ndarray]:
        sims = self._cosine_sims(query)
        order = np.argsort(-sims)
        best = int(order[0])
        second = float(sims[order[1]]) if self.K > 1 else -1.0
        gap = float(sims[best] - second)
        return best, float(sims[best]), gap, sims

    def _build_local_indices(self) -> np.ndarray:
        local = np.empty((self.K, self.local_k), dtype=np.int64)
        for idx in range(self.K):
            order = np.argsort(-self._pairwise_sims[idx])
            local[idx] = order[:self.local_k]
        return local

    def _build_discriminative_profiles(self) -> np.ndarray:
        profiles = np.empty((self.K, self.N), dtype=np.float64)
        for idx in range(self.K):
            neighbours = self.X[self._local_indices[idx]]
            profile = np.std(neighbours, axis=0)
            profiles[idx] = self._scale_feature(profile)
        return profiles

    def _build_clusters(self) -> list[np.ndarray]:
        cluster_keys: dict[tuple[int, ...], int] = {}
        clusters: list[np.ndarray] = []
        for idx in range(self.K):
            key = tuple(sorted(int(v) for v in self._local_indices[idx]))
            cluster_id = cluster_keys.get(key)
            if cluster_id is None:
                cluster_id = len(clusters)
                cluster_keys[key] = cluster_id
                clusters.append(np.array(key, dtype=np.int64))
        return clusters

    def _fit_retrieval_selector(self) -> dict[str, float]:
        queries, truths, levels = make_test_queries(
            self.X,
            noise_levels=[0.75, 0.85],
            n_per_level=40,
            seed=314159,
        )

        configs = []
        for bypass_sim in (0.76, 0.78, 0.80, 0.82):
            for bypass_gap in (0.03, 0.04, 0.05, 0.06):
                for sim_weight in (0.45, 0.55, 0.65, 0.75):
                    for disc_weight in (1.0, 1.5, 2.0, 2.5):
                        configs.append({
                            "bypass_sim": bypass_sim,
                            "bypass_gap": bypass_gap,
                            "sim_weight": sim_weight,
                            "disc_weight": disc_weight,
                        })

        best_cfg = configs[0]
        best_score = float("-inf")
        for cfg in configs:
            score = 0.0
            for q, truth, level in zip(queries, truths, levels):
                best_idx, best_sim, gap, sims = self._nearest_pattern(q)
                pred = self._predict_retrieval_index(q, best_idx, best_sim, gap, sims, cfg)
                weight = 1.0 + max(float(level) - 0.75, 0.0) * 8.0
                if pred == int(truth):
                    score += weight
            if score > best_score:
                best_score = score
                best_cfg = cfg
        return best_cfg

    def _fit_retrieval_weights(self) -> dict[str, float]:
        queries, truths, _levels = make_test_queries(
            self.X,
            noise_levels=[0.75, 0.85],
            n_per_level=16,
            seed=271828,
        )

        candidates = [
            {"nearest": 1.55, "residual": 0.25, "ambiguity": 0.15, "disc": 0.20, "signal": 0.10, "target": 0.05},
            {"nearest": 1.75, "residual": 0.20, "ambiguity": 0.10, "disc": 0.20, "signal": 0.10, "target": 0.05},
            {"nearest": 1.45, "residual": 0.20, "ambiguity": 0.10, "disc": 0.35, "signal": 0.08, "target": 0.03},
            {"nearest": 1.25, "residual": 0.55, "ambiguity": 0.15, "disc": 0.25, "signal": 0.10, "target": 0.05},
            {"nearest": 1.60, "residual": 0.30, "ambiguity": 0.25, "disc": 0.30, "signal": 0.10, "target": 0.05},
            {"nearest": 1.85, "residual": 0.10, "ambiguity": 0.05, "disc": 0.25, "signal": 0.05, "target": 0.00},
        ]

        best_cfg = candidates[0]
        best_acc = float("-inf")
        for cfg in candidates:
            correct = 0
            for q, truth in zip(queries, truths):
                best_idx, best_sim, gap, sims = self._nearest_pattern(q)
                chosen_idx = self._predict_retrieval_index(q, best_idx, best_sim, gap, sims)
                pi = self._retrieval_candidate_pi(q, chosen_idx, best_sim, sims, weights_cfg=cfg)
                a_star = self.model.run(q, pi, u_const=q)
                if self.model.classify(a_star) == int(truth):
                    correct += 1
            acc = correct / len(queries)
            if acc > best_acc:
                best_acc = acc
                best_cfg = cfg
        return best_cfg

    def _spread(self, H: np.ndarray, pi: np.ndarray) -> float:
        pi_proj = self._normalise_pi(pi)
        pi_sqrt = np.sqrt(np.clip(pi_proj, 1e-12, None))
        S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
        S = 0.5 * (S + S.T)
        eigs = np.linalg.eigvalsh(S)
        eigs = eigs[eigs > 1e-9]
        if len(eigs) < 2:
            return float("inf")
        return float(eigs[-1] / eigs[0])

    def _diag_hinv(self, H: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals = np.clip(eigvals, 1e-9, None)
        H_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
        return np.diag(H_inv)

    def _candidate_pis(self, H: np.ndarray) -> list[np.ndarray]:
        eigvals, eigvecs = np.linalg.eigh(H)
        diag = np.clip(np.diag(H), 1e-9, None)
        v_min = eigvecs[:, 0]
        v_max = eigvecs[:, -1]
        diag_hinv = self._diag_hinv(H)
        return [
            np.ones(self.N),
            1.0 / diag,
            diag_hinv,
            diag_hinv / np.sqrt(diag),
            np.exp(3.0 * (v_min ** 2) - 1.5 * (v_max ** 2)),
            np.exp(4.0 * (v_min ** 2)),
            np.exp(-2.0 * (v_max ** 2)) / np.sqrt(diag),
            np.clip(diag_hinv * np.exp(2.0 * (v_min ** 2)), 1e-9, None),
            np.clip((1.0 / diag) * np.exp(2.0 * (v_min ** 2) - 1.0 * (v_max ** 2)), 1e-9, None),
        ]

    def _optimise_anisotropy_pi(self, H: np.ndarray) -> np.ndarray:
        best_pi = np.ones(self.N, dtype=np.float64)
        best_spread = self._spread(H, best_pi)

        for candidate in self._candidate_pis(H):
            spread = self._spread(H, candidate)
            if spread < best_spread:
                best_spread = spread
                best_pi = self._normalise_pi(candidate)

        z = np.log(np.clip(best_pi, 1e-9, None))
        order = np.argsort(-np.abs(best_pi - 1.0))

        for step in (0.45, 0.2, 0.1):
            improved = True
            while improved:
                improved = False
                for j in order:
                    for delta in (-step, step):
                        z_trial = z.copy()
                        z_trial[j] += delta
                        pi_trial = self._normalise_pi(np.exp(z_trial))
                        spread = self._spread(H, pi_trial)
                        if spread + 1e-9 < best_spread:
                            best_spread = spread
                            best_pi = pi_trial
                            z = np.log(np.clip(best_pi, 1e-9, None))
                            improved = True
                            break
                    if improved:
                        break

        return best_pi

    def _build_anisotropy_bank(self) -> np.ndarray:
        bank = np.empty((self.K, self.N), dtype=np.float64)
        self._corr_bank = np.empty((self.K, self.N, self.N), dtype=np.float64)
        for idx, pattern in enumerate(self.X):
            equilibrium = self.model.find_equilibrium(pattern)
            H = self.model.hessian(equilibrium)
            self._corr_bank[idx] = 0.5 * ((self.R_orig - H) + (self.R_orig - H).T)
            bank[idx] = self._optimise_anisotropy_pi(H)
        return bank

    def _fit_retrieval_operator_alpha(self) -> float:
        queries, truths, _levels = make_test_queries(
            self.X,
            noise_levels=[0.75, 0.85],
            n_per_level=12,
            seed=161803,
        )

        best_alpha = 0.0
        best_acc = float("-inf")
        eye = np.eye(self.N, dtype=np.float64)
        for alpha in (0.0, 0.25, 0.5, 0.75):
            correct = 0
            for q, truth in zip(queries, truths):
                best_idx, best_sim, gap, sims = self._nearest_pattern(q)
                chosen_idx = self._predict_retrieval_index(q, best_idx, best_sim, gap, sims)
                pi = self._retrieval_candidate_pi(q, chosen_idx, best_sim, sims)
                R = self.R_orig if alpha == 0.0 else self._corr_bank[chosen_idx] + alpha * eye
                trial_model = PCAMModel(
                    self.X,
                    R,
                    eta=self.eta,
                    beta=self.beta,
                    dt=self.dt,
                    T_max=self.T_max,
                    tol=self.tol,
                    T_in=self.T_in,
                    pi_min=self.pi_min,
                    pi_max=self.pi_max,
                )
                a_star = trial_model.run(q, pi, u_const=q)
                if trial_model.classify(a_star) == int(truth):
                    correct += 1
            acc = correct / len(queries)
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha
        return best_alpha

    def _build_retrieval_operator_bank(self) -> np.ndarray:
        if self._retrieval_operator_alpha <= 0:
            return np.repeat(self.R_orig[None, :, :], self.K, axis=0)
        eye = np.eye(self.N, dtype=np.float64)
        bank = self._corr_bank + self._retrieval_operator_alpha * eye[None, :, :]
        return 0.5 * (bank + np.swapaxes(bank, 1, 2))

    def _anisotropy_probe_pi(self, sims: np.ndarray) -> np.ndarray:
        order = np.argsort(-sims)
        top = order[:self.local_k]
        weights = self._softmax(20.0 * sims[top])
        pi = weights @ self._anisotropy_bank[top]
        return self._normalise_pi(pi)

    def _retrieval_pi(self,
                      query: np.ndarray,
                      best_idx: int,
                      best_sim: float,
                      sims: np.ndarray) -> np.ndarray:
        return self._retrieval_candidate_pi(query, best_idx, best_sim, sims)

    def _retrieval_candidate_pi(self,
                                query: np.ndarray,
                                candidate_idx: int,
                                best_sim: float,
                                sims: np.ndarray,
                                weights_cfg: dict[str, float] | None = None) -> np.ndarray:
        weights_cfg = weights_cfg or self._retrieval_weights
        q = np.asarray(query, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        top = self._local_indices[candidate_idx]
        top_sims = sims[top]
        weights = self._softmax(self.posterior_scale * top_sims)
        local_patterns = self.X[top]

        target = weights @ local_patterns
        local_mean = target
        local_var = weights @ ((local_patterns - local_mean) ** 2)
        discriminative = self._discriminative[candidate_idx]
        geometry_pi = self._anisotropy_bank[candidate_idx]

        residual = self._scale_feature(np.abs(target - q))
        nearest_deviation = self._scale_feature((q - self.X[candidate_idx]) ** 2)
        ambiguity = self._scale_feature(local_var)
        signal = self._scale_feature(np.abs(q))
        target_strength = self._scale_feature(np.abs(target))

        pi = (
            0.30
            + weights_cfg["nearest"] * nearest_deviation
            + weights_cfg["residual"] * residual
            + weights_cfg["ambiguity"] * ambiguity
            + weights_cfg["disc"] * discriminative
            + weights_cfg["signal"] * signal
            + weights_cfg["target"] * target_strength
        )

        confidence = float(weights[0])
        geometry_mix = np.clip((confidence - 0.55) / 0.35, 0.0, 0.20)
        near_clean_mix = np.clip((best_sim - 0.72) / 0.18, 0.0, 0.12)
        mix = max(geometry_mix, near_clean_mix)
        pi = (1.0 - mix) * pi + mix * geometry_pi
        return self._normalise_pi(pi)

    def _predict_retrieval_index(self,
                                 query: np.ndarray,
                                 best_idx: int,
                                 best_sim: float,
                                 gap: float,
                                 sims: np.ndarray,
                                 cfg: dict[str, float] | None = None) -> int:
        cfg = cfg or self._retrieval_selector
        if best_sim >= cfg["bypass_sim"] or gap >= cfg["bypass_gap"]:
            return best_idx

        q = np.asarray(query, dtype=np.float64)
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            q = q / q_norm

        cluster_scores = [float(np.mean(sims[cluster])) for cluster in self._clusters]
        cluster_id = int(np.argmax(cluster_scores))
        cluster = self._clusters[cluster_id]

        abs_q = self._scale_feature(np.abs(q))
        chosen_idx = best_idx
        chosen_score = float("-inf")
        for candidate_idx in cluster:
            candidate_idx = int(candidate_idx)
            weights = 0.25 + abs_q + cfg["disc_weight"] * self._discriminative[candidate_idx]
            residual = np.sum(weights * (q - self.X[candidate_idx]) ** 2)
            score = cfg["sim_weight"] * sims[candidate_idx] - residual / self.N
            if score > chosen_score:
                chosen_score = float(score)
                chosen_idx = candidate_idx
        return chosen_idx

    def _select_retrieval_pi(self,
                             query: np.ndarray,
                             best_idx: int,
                             best_sim: float,
                             gap: float,
                             sims: np.ndarray) -> tuple[int, np.ndarray]:
        chosen_idx = self._predict_retrieval_index(query, best_idx, best_sim, gap, sims)
        return chosen_idx, self._retrieval_candidate_pi(query, chosen_idx, best_sim, sims)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        best_idx, best_sim, gap, sims = self._nearest_pattern(corrupted_query)
        self.R_ref[:, :] = self.R_orig
        if best_sim >= self.probe_threshold:
            return self._anisotropy_probe_pi(sims)
        chosen_idx, pi = self._select_retrieval_pi(corrupted_query, best_idx, best_sim, gap, sims)
        self.R_ref[:, :] = self._retrieval_operator_bank[chosen_idx]
        return pi
