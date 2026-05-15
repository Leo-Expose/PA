# Dashboard Design

Local dashboard for the `PA/` copy of the audited P-04 bench.

## Presets

- `Quick Test`: seeds `42 101`, noise `[0.72, 0.84]`, fast local high-noise smoke run
- `Full Evaluation`: seeds `42 101 202 303 404`, audited public profile
- `Stress Test`: seeds `503 1009 9999`

## Scorecard Targets

- retrieval full marks at `Δ >= 0.08`
- anisotropy full marks at `5x`
- min `Δ < 0` is a failure condition
- min spread reduction `<= 1.0x` is a failure condition

## Table Columns

- seed id
- direct classify accuracy
- baseline retrieval accuracy
- agent retrieval accuracy
- delta accuracy
- baseline spread
- agent spread
- spread reduction

## Charts

- per-seed `Δ` accuracy bars
- per-seed spread-reduction bars
- baseline vs agent spread curves

## Notes

- The dashboard reflects the local `PA/` quick preset, not the canonical audited quick script from the upstream bench.
- The full run remains the authoritative local reproduction of the public audited profile.
