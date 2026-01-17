# Bubble-Overlap Networks & Systemic Risk Forecasting

This repository contains research code for building **dynamic bubble-overlap networks**, optional **FRM (quantile-regression) networks**, and **TGNN-based forecasting** of node-level risk proxies (e.g., eigenvector centrality).

## Quickstart (Windows / PowerShell)

### 1) Environment
Activate the existing venv (or create a new one) and install dependencies:
```powershell
cd c:\Users\Radu\VSCode\StochasticDominanceBubbles
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

CPU-only alternative:
```powershell
python -m pip install -r requirements-lite.txt
python -m pip install -e .
```

### 2) Run the pipeline
Minimal (fast) run:
```powershell
python scripts/run_pipeline.py --skip-temporal --skip-centrality
```

Full rebuild (creates artifacts in `results/` and `figures/`):
```powershell
python scripts/run_pipeline.py --run-frm --start-date 2020-01-01
```

Run TGNN explicitly:
```powershell
python scripts/run_tgnn.py --mode bubble --epochs 50 --model gconvgru
```

### 3) Build the paper PDF
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_paper.ps1
```
Output: `documents/build/main.pdf`

## Repo structure
```
src/                  # importable research modules (active pipeline)
scripts/              # thin CLIs + build tooling
data/ro/              # Romanian market inputs (Excel)
data/stoxx600/        # STOXX600 inputs (Excel)
figures/              # generated plots (artifacts; not tracked)
results/              # generated tables/pkl (artifacts; not tracked)
documents/            # LaTeX paper + build output
legacy_sd/            # isolated stochastic dominance code (not in active pipeline)
notebooks/            # exploratory notebooks
```

## Artifacts policy
- `figures/` and `results/` are treated as **generated artifacts** and are **not tracked** in git (kept via `.gitkeep`).
- Intermediate `*.pkl` produced by runs is not tracked.

## Stochastic Dominance (SD)
SD is intentionally **removed from the active pipeline** for now and kept isolated in `legacy_sd/`.
See `legacy_sd/README.md`.
