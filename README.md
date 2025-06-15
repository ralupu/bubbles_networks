# Dynamic Bubble-Overlap Networks and Systemic Risk Forecasting

This repository provides the complete codebase for empirical research on **financial bubble contagion** and **dynamic network-based systemic risk**.

---

## Overview

The project analyzes **dynamic financial networks** built from **overlapping bubble periods** in time series of asset prices.  
The main pipeline includes:

- **Bubble detection and descriptive analytics** (by asset)
- **Construction of daily, time-evolving networks** based on bubble overlaps
- **Visualization and temporal centrality analysis** of the resulting networks
- **Temporal GNN (TGNN) forecasting** of node-level risk metrics (eigenvector centrality, ΔCoVaR, etc.)
- **Flexible model configuration via command-line arguments**

> **Note:**  
> While this codebase is a platform for network-based risk analysis, **stochastic dominance and related distributional risk tests are not (yet) implemented.**  
> For purely distributional or comparative risk analysis, see [future extensions](#future-extensions).

---

## Directory Structure

```
repo-root/
│
├── descriptive_bubbles.py           # Bubble detection/descriptive analysis
├── network_analysis.py              # Dynamic network construction (daily)
├── network_simultaneous_bubbles.py  # Bubble overlap graph construction
├── network_leads_laggards.py        # Bubble timing & propagation analysis
├── chart_overlapping_bubbles.py     # Visualization of bubble phases
├── centrality_analysis_module.py    # Computes and visualizes network centralities
├── tgnn_forecasting_module.py       # TGNN (GConvGRU/A3TGCN/EvolveGCN) risk forecasting
│
├── temporal_graphs.pkl              # Serialized daily temporal network objects
├── figures/                         # All output charts (networks, forecasts)
├── data/                            # Raw and processed financial data
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

---

## Quickstart

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```
> For GPU acceleration, ensure you have a CUDA-enabled PyTorch  
> (see [PyTorch install instructions](https://pytorch.org/get-started/locally/)).

---

### 2. **Run the Network Pipeline**

Run the scripts in sequence:
```bash
python descriptive_bubbles.py
python network_analysis.py
python network_simultaneous_bubbles.py
python network_leads_laggards.py
python chart_overlapping_bubbles.py
python centrality_analysis_module.py
```
This generates `temporal_graphs.pkl` and visual outputs in `figures/`.

---

### 3. **TGNN Risk Forecasting**

Run the GNN forecasting script with flexible options:
```bash
python tgnn_forecasting_module.py --hidden-size 128 --epochs 100 --model a3tgcn --lookback 5
```
- **--model**: Choose `gconvgru`, `a3tgcn`, or `evolvegcn`
- **--lookback**: Only used for `a3tgcn`; set to window size for temporal features

Model performance charts will be saved in `figures/` with filenames encoding your parameter choices.

---

## Scientific Purpose

- **Map and visualize the evolution of systemic risk** via daily bubble-overlap networks.
- **Quantify how bubble dynamics impact the centrality (importance) of firms** in financial contagion networks.
- **Benchmark temporal GNNs** for their ability to forecast node-level risk metrics, as a potential early-warning tool for regulators or risk managers.

---

## Future Extensions

- **Stochastic dominance and distributional analysis** comparing risk measures (e.g., ΔCoVaR, centrality) during bubble vs. non-bubble periods.
- **Motif-based and graph resilience metrics** to further analyze market fragility.
- **Integration of additional financial features or alternative graph constructions** (correlation, Granger, etc.)

---

## Credits & Citation

If using this code for research, please cite:

```
@article{YOURPAPER2025,
  title={Dynamic Bubble Networks and Systemic Risk Forecasting},
  author={Radu et al.},
  journal={Working paper / submitted},
  year={2025}
}
```

---

## Contact

For questions, contact [Your Name](mailto:your.email@example.com).
