"""Descriptive (data-section) statistics for FRM networks over time.

This module reads the FRM temporal snapshot pickle (list of (date, PyG Data)) and
computes network-level metrics per snapshot, then aggregates them across time.

Export format requirement:
- LaTeX output contains ONLY a `tabular` environment with booktabs (no table/caption/label).
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_numpy_edge_index(edge_index) -> np.ndarray:
    if edge_index is None:
        return np.zeros((2, 0), dtype=int)
    try:
        # torch tensor
        return edge_index.detach().cpu().numpy().astype(int)
    except Exception:
        return np.asarray(edge_index, dtype=int)


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: float, *, ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{float(x):.{ndigits}f}"


def compute_frm_snapshot_metrics(pyg) -> Dict[str, float]:
    """Compute network-level metrics for a single directed FRM snapshot.

    Metrics:
    - density: m / (n*(n-1)) for directed graphs (excluding self-loops)
    - avg_in_degree: mean in-degree across nodes
    - avg_out_degree: mean out-degree across nodes
    - degree_dispersion_std: std of total degree (in+out) across nodes
    """

    edge_index = _to_numpy_edge_index(getattr(pyg, "edge_index", None))
    m = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0

    n = getattr(pyg, "num_nodes", None)
    if n is None:
        n = int(edge_index.max()) + 1 if m else 0
    n = int(n)

    if n <= 1:
        return {
            "density": 0.0,
            "avg_in_degree": 0.0,
            "avg_out_degree": 0.0,
            "degree_dispersion_std": 0.0,
        }

    if m == 0:
        indeg = np.zeros((n,), dtype=float)
        outdeg = np.zeros((n,), dtype=float)
    else:
        src = edge_index[0, :].astype(int)
        dst = edge_index[1, :].astype(int)
        outdeg = np.bincount(src, minlength=n).astype(float)
        indeg = np.bincount(dst, minlength=n).astype(float)

    density = float(m / (n * (n - 1)))
    avg_in = float(indeg.mean())
    avg_out = float(outdeg.mean())
    deg_total = indeg + outdeg
    deg_std = float(deg_total.std(ddof=0))

    return {
        "density": density,
        "avg_in_degree": avg_in,
        "avg_out_degree": avg_out,
        "degree_dispersion_std": deg_std,
    }


def compute_frm_metrics_over_time(
    frm_graphs: Sequence[Tuple[object, object]],
) -> pd.DataFrame:
    """Compute per-snapshot FRM metrics as a dataframe indexed by date."""

    rows: List[Dict[str, object]] = []
    for d, pyg in list(frm_graphs):
        rows.append({"date": pd.to_datetime(d), **compute_frm_snapshot_metrics(pyg)})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date").reset_index(drop=True)
    return df


def write_frm_temporal_stats_table(
    *,
    frm_graphs_pkl: str,
    out_tex: str,
) -> Dict[str, str]:
    """Write a tabular-only LaTeX table of FRM network stats aggregated over time."""

    if not os.path.exists(frm_graphs_pkl):
        raise FileNotFoundError(f"FRM graphs pickle not found: {frm_graphs_pkl}")

    with open(frm_graphs_pkl, "rb") as f:
        frm_graphs = pickle.load(f)
    if not isinstance(frm_graphs, list):
        raise TypeError(f"Expected list of (date, Data) in {frm_graphs_pkl}, got {type(frm_graphs)}")

    df = compute_frm_metrics_over_time(frm_graphs)
    if df.empty:
        raise RuntimeError(f"No FRM snapshots found in {frm_graphs_pkl}")

    metrics = [
        ("density", "Network density"),
        ("avg_in_degree", "Average in-degree"),
        ("avg_out_degree", "Average out-degree"),
        ("degree_dispersion_std", "Degree dispersion (std of total degree)"),
    ]

    summary_rows = []
    for key, label in metrics:
        s = pd.to_numeric(df[key], errors="coerce")
        vals = s.to_numpy(dtype=float, copy=True)
        finite = np.isfinite(vals)
        if not finite.any():
            mean = std = vmin = vmax = float("nan")
        else:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        summary_rows.append({"Metric": label, "Mean": mean, "Std": std, "Min": vmin, "Max": vmax})

    out = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{p{0.48\\textwidth} r r r r}\n")
        f.write("\\toprule\n")
        f.write("Metric & Mean & Std & Min & Max \\\\\n")
        f.write("\\midrule\n")
        for _, r in out.iterrows():
            metric = str(r["Metric"]).replace("&", "\\&")
            f.write(
                f"{metric} & "
                f"{_fmt(_safe_float(r['Mean']), ndigits=4)} & "
                f"{_fmt(_safe_float(r['Std']), ndigits=4)} & "
                f"{_fmt(_safe_float(r['Min']), ndigits=4)} & "
                f"{_fmt(_safe_float(r['Max']), ndigits=4)} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    return {
        "frm_stats_snapshots": str(int(len(df))),
        "frm_stats_date_start": str(pd.to_datetime(df["date"].min()).date()),
        "frm_stats_date_end": str(pd.to_datetime(df["date"].max()).date()),
    }
