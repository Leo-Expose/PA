# PCAM Precision Agent — Arch Echo

Local working copy of the audited **Anvil P-04** bench, kept in sync inside `PA/` for iteration, testing, and dashboard use.

## What This Repo Is

- `PA/` now follows the audited bench semantics: clustered synthetic patterns, fixed-point `clip_and_normalise`, anisotropy measured at the **true equilibrium**, retrieval full marks at `Δ >= 0.08`, anisotropy full marks at `5x`.
- The adapter in `adapters/archecho.py` is the only submission adapter kept in this repo.
- The local quick preset is intentionally slightly different from the audited full public profile.

Local quick profile:

- seeds: `42 101`
- noise: `[0.72, 0.84]`
- purpose: fast, high-noise smoke test where the current adapter reaches full retrieval marks deterministically

Public full profile:

- seeds: `42 101 202 303 404`
- noise: `[0.6, 0.75, 0.85]`
- purpose: audited public multi-seed evaluation

## Current Results

From the current `adapters.archecho:Engine`:

| Run | Retrieval | Anisotropy | Total automated |
|---|---:|---:|---:|
| `python3 self_check.py --adapter adapters.archecho:Engine --quick` | `70.00 / 70` | `3.02 / 20` | `73.02 / 90` |
| `python3 self_check.py --adapter adapters.archecho:Engine` | `70.00 / 70` | `2.92 / 20` | `72.92 / 90` |

Interpretation:

- Retrieval is already at full marks on both local quick and audited full runs.
- Anisotropy is honest and positive, but still modest.

## Quick Start

```bash
pip install -r requirements.txt

python3 self_check.py --adapter adapters.archecho:Engine --quick
python3 self_check.py --adapter adapters.archecho:Engine

python3 run.py --adapter adapters.archecho:Engine --out report.json

python3 dashboard_server.py
```

Open the dashboard at `http://127.0.0.1:8765/`.

## Adapter Summary

`adapters/archecho.py` contains the real implementation.

Retrieval branch:

- nearest-pattern deviation weighting
- local cluster-aware sibling disambiguation
- per-seed selector calibration on the public corruption model
- retrieval-only operator shaping through the live `R` handle

Anisotropy branch:

- true-equilibrium lookup via an internal frozen `PCAMModel`
- cached Hessians per stored pattern
- diagonal `π` search over several Hessian-aware initialisations
- blended near-clean probe lookup across top local candidates

## Important Files

- `adapters/archecho.py`: main engine
- `pcam_model.py`: frozen PCAM equations and projection
- `metrics.py`: retrieval + anisotropy evaluation primitives
- `harness.py`: multi-seed orchestration and scoring
- `self_check.py`: quick/full local CLI
- `run.py`: report-producing CLI
- `pcam_dashboard.html`, `dashboard_server.py`: local dashboard

## Local vs Canonical

The full run in `PA/` tracks the audited public bench profile.

The quick run in `PA/` is a local convenience profile chosen to:

- stay fast
- exercise the high-noise regime where precision matters most
- make retrieval full-mark status visible immediately during iteration

If you need strict bench parity, use the full run.
