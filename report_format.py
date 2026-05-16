"""Shared human-readable formatter for P-04 reports."""
from __future__ import annotations

from typing import Any

from harness import ANISOTROPY_FULL_AT, RETRIEVAL_FULL_AT


def print_report(report: dict[str, Any], total_ms: float) -> None:
    cfg = report["config"]
    agg = report["aggregated"]
    sc = report["score"]

    print()
    print("ANVIL · P-04 · PCAM Precision Agent — Self-Check")
    print("=" * 72)
    print(f"  total wall time          {total_ms:>10.1f} ms")
    print(f"  seeds                    {agg['n_seeds']:>10d}")
    print(f"  stored patterns (K)      {cfg['K']:>10d}")
    print(f"  state dim (N)            {cfg['N']:>10d}")
    print(f"  noise levels             {cfg['noise_levels']}")
    print()
    print("  PER-SEED   ─ retrieval ─────────────       ── anisotropy ──")
    print("  seed     direct  Π=I    agent    Δ          base   agent   reduction")
    print("  " + "-" * 70)
    for r in report["per_seed"]:
        flag = "✓" if r["dynamics_adds_value"] else "✗"
        print(f"  {r['seed']:>4}    {r['direct_classify_acc']:.3f}  "
              f"{r['baseline_acc']:.3f}  {r['agent_acc']:.3f}  "
              f"{r['delta']:+.3f} {flag}    "
              f"{r['baseline_spread']:>6.2f}  {r['agent_spread']:>6.2f}  "
              f"{r['spread_reduction']:>5.2f}×")
    print()
    print("  AGGREGATED                                  VALUE")
    print("  " + "-" * 70)
    print(f"  mean Δ accuracy (over seeds)               {agg['mean_delta']:+.3f}")
    print(f"  min  Δ accuracy (worst seed)               {agg['min_delta']:+.3f}")
    print(f"  mean spread reduction                      {agg['mean_reduction']:>6.2f}×")
    print(f"  min  spread reduction                      {agg['min_reduction']:>6.2f}×")
    print(f"  dynamics-adds-value pass rate              {agg['dynamics_gate_pass_rate']:.0%}")
    print()
    print("  SCORE (automated, max 90)                  POINTS")
    print("  " + "-" * 70)
    print(f"  retrieval     (max 70)                     {sc['retrieval_pts']:>6.2f}")
    print(f"  anisotropy    (max 20)                     {sc['anisotropy_pts']:>6.2f}")
    print(f"  code quality  (max 10)                     (manual)")
    print(f"  TOTAL AUTOMATED                            {sc['total_automated']:>6.2f}  / 90")
    print()
    if sc["notes"]:
        print("  NOTES")
        for note in sc["notes"]:
            print(f"    · {note}")
        print()


def print_advice(report: dict[str, Any]) -> None:
    agg = report["aggregated"]
    delta = agg["mean_delta"]
    spread = agg["mean_reduction"]

    full_delta = RETRIEVAL_FULL_AT
    half_delta = 0.5 * full_delta

    if delta <= 0:
        print("  Mean Δ <= 0 — your agent does not beat Π=I on average.")
    elif delta < 0.4 * half_delta:
        print(f"  Mean Δ {delta:+.3f} is small. The agent helps, but not sharply yet.")
    elif delta < full_delta:
        print(
            f"  Retrieval is solid (Δ={delta:+.3f}), but still below the full-mark "
            f"threshold of Δ >= {full_delta:.2f}."
        )
    else:
        print(f"  Retrieval is at full marks on this run (Δ={delta:+.3f}).")

    full_spread = ANISOTROPY_FULL_AT
    if spread <= 1.0:
        print("  Spread reduction is at or below baseline. The anisotropy branch needs work.")
    elif spread < 0.2 * full_spread:
        print(
            f"  Anisotropy is improving modestly ({spread:.2f}×). There is still "
            f"substantial headroom before the {full_spread:.0f}× full-mark threshold."
        )
    elif spread < full_spread:
        print(
            f"  Anisotropy is meaningful ({spread:.2f}×), but still below the "
            f"{full_spread:.0f}× full-mark threshold."
        )
    else:
        print(f"  Anisotropy is at full marks on this run ({spread:.2f}×).")
    print()
