# Legacy: Stochastic Dominance (SD)

This folder contains SD-related research code kept for reference and potential future reuse.

SD is **not part of the active pipeline** and must not be imported by `src/` or `scripts/`.

## What’s here
- `legacy_sd/previous_main_with_SD.py`: legacy SD workflow script (ported from earlier experiments).

## How to run
From the repo root (after activating `.venv` and installing dependencies):
```powershell
python legacy_sd/previous_main_with_SD.py
```

## Inputs
This script reads the Romanian (RO) Excel inputs from:
- `data/ro/ResultResults_ro_bet_bubbles.xlsx`
- `data/ro/ResultResults_ro_bet_covars.xlsx`

## Outputs
Outputs are written to:
- `legacy_sd/output/results_S1.xlsx`
- `legacy_sd/output/results_S2.xlsx`
- `legacy_sd/output/results_S3.xlsx`

These outputs are considered **artifacts** (not tracked in git).
