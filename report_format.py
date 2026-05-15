"""
Shared human-readable formatter for P-04 reports.

Used by both `self_check.py` and `run.py` so the two commands present
identical output. Anything in `report` produced by `harness.run_multi`
flows through `print_report`.
"""
from __future__ import annotations

from typing import Any


def print_report(report: dict[str, Any], total_ms: float) -> None:
    cfg = report["config"]
    agg = report["aggregated"]
    sc = report["score"]

    K = cfg["K"]
    N = cfg["N"]
    noise_levels = cfg["noise_levels"]

    print()
    print("ANVIL · P-04 · PCAM Precision Agent — Self-Check")
    print("=" * 62)
    print(f"  total wall time          {total_ms:>10.1f} ms")
    print(f"  seeds                    {agg['n_seeds']:>10d}")
    print(f"  stored patterns (K)      {K:>10d}")
    print(f"  state dim (N)            {N:>10d}")
    print(f"  noise levels             {noise_levels}")
    print()
    print("  PER-SEED  ─ retrieval ─       ── anisotropy ──")
    print("  seed      Π=I      agent  Δ     base    agent  ratio")
    print("  " + "-" * 58)
    for r in report["per_seed"]:
        print(f"  {r['seed']:>4}     {r['baseline_accuracy']:.3f}    "
              f"{r['agent_accuracy']:.3f}  {r['delta']:+.3f}  "
              f"{r['spread_baseline']:>6.2f}  {r['spread_agent']:>6.2f}  "
              f"{r['spread_reduction']:>5.2f}×")
    print()
    print("  AGGREGATED                       VALUE")
    print("  " + "-" * 58)
    print(f"  mean Δ accuracy (over seeds)    {agg['mean_delta']:+.3f}")
    print(f"  min  Δ accuracy (worst seed)    {agg['min_delta']:+.3f}")
    print(f"  mean spread reduction           {agg['mean_spread']:>6.2f}×")
    print(f"  min  spread reduction           {agg['min_spread']:>6.2f}×")
    print()
    print("  SCORE (automated, max 90)         POINTS")
    print("  " + "-" * 58)
    print(f"  retrieval     (max 70)            {sc['retrieval_pts']:>6.2f}")
    print(f"  anisotropy    (max 20)            {sc['anisotropy_pts']:>6.2f}")
    print(f"  code quality  (max 10)            (manual)")
    print(f"  TOTAL AUTOMATED                   {sc['total_automated']:>6.2f}  / 90")
    print()


def print_advice(report: dict[str, Any]) -> None:
    """Optional commentary printed by self_check after the table."""
    agg = report["aggregated"]
    delta = agg["mean_delta"]
    spread = agg["mean_spread"]
    min_d = agg["min_delta"]

    if delta <= 0:
        print("  ⚠  Mean Δ ≤ 0 — your agent does not beat Π=I on average. "
              "Zero on retrieval.")
    elif delta < 0.02:
        print("  ▸  Mean Δ is small. The agent is helping a bit but not "
              "principled. Aim for Δ ≥ 0.05.")
    else:
        print("  ✓  Solid retrieval gain on average.")

    if min_d < 0:
        print("  ⚠  Min Δ < 0 — your agent regresses on at least one seed. "
              "Retrieval score halved.")
    if spread < 2.0:
        print("  ▸  Spread reduction near baseline. Try Hessian-aware design "
              "(see README hints).")
    elif spread < 10.0:
        print("  ✓  Anisotropy improving — log-scaled toward 10×.")
    else:
        print("  ✓✓ Excellent anisotropy reduction.")
    print()
