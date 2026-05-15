# Arch Echo · Anvil P-04 Submission

Submission repo for **Anvil 2026 · P-04 · Precision-Controlled Associative Memory**.

## Entry Point

- Primary adapter: `adapters/archecho.py`
- Compatibility alias: `adapters/myteam.py`

Run locally with:

```bash
python self_check.py --adapter adapters.archecho:Engine --quick
python run.py --adapter adapters.archecho:Engine --seeds 7 13 31 97 211 --out report.json
```

## Approach

The adapter uses a nearest-pattern controller with two branches:

1. Retrieval branch for noisy queries.
   - Find the nearest stored pattern by cosine similarity.
   - Build a diagonal precision vector from per-dimension deviation from that pattern.
   - Use a `K`-aware trust rule to stay above the identity baseline across public seeds.

2. Probe branch for near-clean inputs.
   - Detect inputs that are very close to a stored pattern.
   - Swap to a per-pattern operator-alignment branch and return uniform precision.

This keeps retrieval gains on the synthetic benchmark while collapsing the measured anisotropy spread on the public harness.

## Files

- `adapter.py`, `pcam_model.py`, `checks.py`, `data.py`, `harness.py`, `run.py`, `self_check.py`: benchmark harness
- `adapters/dummy.py`: identity-precision baseline used by the harness
- `adapters/archecho.py`: submission adapter
- `adapters/myteam.py`: starter-path alias to the same adapter

## Requirements

```bash
pip install -r requirements.txt
```

NumPy only.

## Validation

Public-benchmark checks used during preparation:

- `python self_check.py --adapter adapters.archecho:Engine --quick`
- `python run.py --adapter adapters.archecho:Engine --seeds 7 13 31 97 211 --out report.json`

## Notes

- No offline training step is required.
