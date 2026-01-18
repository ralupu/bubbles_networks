"""Time-aware similarity metrics between bubble and FRM dynamic networks.

This module compares two temporal graph sequences (lists of (date, PyG Data)) by:
- aligning dates (default: nearest prior FRM date for each bubble date), and
- computing node/edge similarity metrics per aligned date.

The comparison is designed for paper-facing outputs:
- a time-series CSV under `results/compare/` (ignored),
- a summary LaTeX table under `documents/tables/` (tracked), and
- a readability-first time-series figure under `documents/figures/` (tracked).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlignmentSummary:
    """Diagnostics about date alignment decisions."""

    method: str
    bubble_dates: int
    frm_dates: int
    aligned_pairs: int
    exact_matches: int
    shifted_matches: int
    max_shift_days: int


def _safe_int(x: object, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: object) -> float:
    try:
        v = float(x)
        if np.isnan(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


def _edge_weights_from_pyg(pyg) -> np.ndarray:
    w = getattr(pyg, "weight", None)
    if w is None:
        w = getattr(pyg, "edge_weight", None)
    if w is None:
        edge_index = getattr(pyg, "edge_index", None)
        m = int(edge_index.shape[1]) if edge_index is not None else 0
        return np.ones((m,), dtype=float)
    try:
        return np.asarray(w.detach().cpu().numpy(), dtype=float)
    except Exception:
        return np.asarray(w, dtype=float)


def align_by_nearest_prior_date(
    bubble_dates: Sequence[pd.Timestamp],
    frm_dates: Sequence[pd.Timestamp],
) -> Tuple[List[Tuple[pd.Timestamp, pd.Timestamp, int]], AlignmentSummary]:
    """Align bubble dates to the nearest prior FRM date (<= bubble date).

    Returns a list of tuples: (bubble_date, frm_date, shift_days).
    """

    b = [pd.to_datetime(d) for d in bubble_dates]
    f = sorted([pd.to_datetime(d) for d in frm_dates])
    if not f:
        return [], AlignmentSummary("nearest_prior", len(b), 0, 0, 0, 0, 0)

    aligned: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    exact = 0
    shifted = 0
    max_shift = 0

    # two-pointer scan
    j = 0
    for bd in b:
        while j + 1 < len(f) and f[j + 1] <= bd:
            j += 1
        if f[j] > bd:
            continue
        fd = f[j]
        shift_days = int((bd.normalize() - fd.normalize()).days)
        aligned.append((bd, fd, shift_days))
        if shift_days == 0:
            exact += 1
        else:
            shifted += 1
        max_shift = max(max_shift, shift_days)

    return aligned, AlignmentSummary(
        method="nearest_prior",
        bubble_dates=len(b),
        frm_dates=len(f),
        aligned_pairs=len(aligned),
        exact_matches=exact,
        shifted_matches=shifted,
        max_shift_days=int(max_shift),
    )


def _restrict_pyg_to_indices(pyg, keep_indices: List[int]):
    """Induce a node-subgraph on a PyG Data object by keeping the given node indices."""

    import torch
    from torch_geometric.data import Data

    keep = sorted(set(int(i) for i in keep_indices))
    if not keep:
        return Data(edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=0)

    old_to_new = {old: new for new, old in enumerate(keep)}

    edge_index = getattr(pyg, "edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif hasattr(edge_index, "detach"):
        edge_index = edge_index.detach().cpu()

    if edge_index.numel() == 0:
        new_edge_index = torch.empty((2, 0), dtype=torch.long)
        new_weight = torch.empty((0,), dtype=torch.float)
    else:
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        keep_mask = [(s in old_to_new) and (t in old_to_new) for s, t in zip(src, dst)]
        new_src = [old_to_new[s] for s, ok in zip(src, keep_mask) if ok]
        new_dst = [old_to_new[t] for t, ok in zip(dst, keep_mask) if ok]
        new_edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        w = _edge_weights_from_pyg(pyg)
        new_weight = torch.tensor([float(w[i]) for i, ok in enumerate(keep_mask) if ok], dtype=torch.float)

    x = getattr(pyg, "x", None)
    new_x = None
    if x is not None:
        x_cpu = x.detach().cpu() if hasattr(x, "detach") else x
        new_x = x_cpu[keep]

    out = Data(edge_index=new_edge_index, num_nodes=len(keep))
    if new_x is not None:
        out.x = new_x
    out.weight = new_weight
    return out


def restrict_temporal_graphs_to_firms(
    graphs: Sequence[Tuple[pd.Timestamp, object]],
    firm_list: Sequence[str],
    *,
    keep_firms: Sequence[str],
) -> Tuple[List[Tuple[pd.Timestamp, object]], List[str]]:
    """Restrict temporal graphs to a set of firms (by firm names)."""

    firm_to_idx = {str(f): i for i, f in enumerate([str(x) for x in firm_list])}
    keep = [firm_to_idx[str(f)] for f in keep_firms if str(f) in firm_to_idx]
    keep_names = [str(firm_list[i]) for i in keep]
    out = [(pd.to_datetime(d), _restrict_pyg_to_indices(g, keep)) for d, g in graphs]
    return out, keep_names


def _pyg_to_edge_table(pyg, firms: Sequence[str]) -> pd.DataFrame:
    edge_index = getattr(pyg, "edge_index", None)
    if edge_index is None:
        return pd.DataFrame(columns=["u", "v", "weight"])
    edges = edge_index.detach().cpu().numpy().T.astype(int) if hasattr(edge_index, "detach") else np.asarray(edge_index).T
    w = _edge_weights_from_pyg(pyg)
    rows = []
    for k, (u, v) in enumerate(edges):
        uu = str(firms[int(u)]) if 0 <= int(u) < len(firms) else f"Unknown_{u}"
        vv = str(firms[int(v)]) if 0 <= int(v) < len(firms) else f"Unknown_{v}"
        ww = float(w[k]) if k < len(w) else 1.0
        rows.append((uu, vv, ww))
    return pd.DataFrame(rows, columns=["u", "v", "weight"])


def _eigenvector_centrality_series(pyg, firms: Sequence[str]) -> pd.Series:
    edge_df = _pyg_to_edge_table(pyg, firms)
    G = nx.Graph()
    for f in firms:
        G.add_node(str(f))
    for _, r in edge_df.iterrows():
        if str(r["u"]) == str(r["v"]):
            continue
        G.add_edge(str(r["u"]), str(r["v"]), weight=float(abs(r["weight"])))
    if G.number_of_nodes() == 0:
        return pd.Series(dtype=float)
    if G.number_of_edges() == 0:
        return pd.Series({n: 0.0 for n in G.nodes()}, dtype=float)
    try:
        ev = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        ev = {n: 0.0 for n in G.nodes()}
    s = pd.Series({str(k): float(v) for k, v in ev.items()}, dtype=float)
    return s.reindex(sorted(s.index), fill_value=0.0)


def _rank_similarity(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    keys = sorted(set(a.index) | set(b.index))
    if not keys:
        return float("nan"), float("nan")
    ra = a.reindex(keys, fill_value=0.0).rank(ascending=False, method="average")
    rb = b.reindex(keys, fill_value=0.0).rank(ascending=False, method="average")
    if int(ra.nunique(dropna=False)) <= 1 or int(rb.nunique(dropna=False)) <= 1:
        return float("nan"), float("nan")
    try:
        from scipy.stats import kendalltau, spearmanr

        sp = float(spearmanr(ra.to_numpy(), rb.to_numpy()).correlation)
        kt = float(kendalltau(ra.to_numpy(), rb.to_numpy()).correlation)
    except Exception:
        sp = float(ra.corr(rb, method="spearman"))
        kt = float("nan")
    return sp, kt


def _topk(series: pd.Series, k: int) -> List[str]:
    if series.empty:
        return []
    items = series.sort_values(ascending=False)
    return [str(x) for x in items.head(int(k)).index.tolist()]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _topm_edges(edge_df: pd.DataFrame, m: int) -> pd.DataFrame:
    if edge_df.empty:
        return edge_df.copy()
    df = edge_df.copy()
    df["w_abs"] = df["weight"].astype(float).abs()
    df = df.sort_values(["w_abs", "u", "v"], ascending=[False, True, True])
    return df.head(int(m)).drop(columns=["w_abs"])


def compute_bubble_vs_frm_similarity_timeseries(
    *,
    bubble_graphs: Sequence[Tuple[pd.Timestamp, object]],
    bubble_firms: Sequence[str],
    frm_graphs: Sequence[Tuple[pd.Timestamp, object]],
    frm_firms: Sequence[str],
    align_method: str = "nearest_prior",
    topk_values: Sequence[int] = (5, 10),
    topm_edges: int = 50,
) -> Tuple[pd.DataFrame, AlignmentSummary]:
    """Compute per-date similarity metrics after date alignment."""

    b_dates = [pd.to_datetime(d) for d, _ in bubble_graphs]
    f_dates = [pd.to_datetime(d) for d, _ in frm_graphs]

    if align_method != "nearest_prior":
        raise ValueError(f"Unsupported align_method: {align_method}")

    align_pairs, summary = align_by_nearest_prior_date(b_dates, f_dates)
    bubble_by_date = {pd.to_datetime(d): g for d, g in bubble_graphs}
    frm_by_date = {pd.to_datetime(d): g for d, g in frm_graphs}

    rows = []
    for bd, fd, shift_days in align_pairs:
        gb = bubble_by_date.get(pd.to_datetime(bd))
        gf = frm_by_date.get(pd.to_datetime(fd))
        if gb is None or gf is None:
            continue

        cb = _eigenvector_centrality_series(gb, bubble_firms)
        cf = _eigenvector_centrality_series(gf, frm_firms)

        # restrict to common firms for rank-based comparisons
        common = sorted(set(cb.index) & set(cf.index))
        cb_c = cb.reindex(common, fill_value=0.0)
        cf_c = cf.reindex(common, fill_value=0.0)

        sp, kt = _rank_similarity(cb_c, cf_c)
        out = {
            "date_bubble": pd.to_datetime(bd).strftime("%Y-%m-%d"),
            "date_frm": pd.to_datetime(fd).strftime("%Y-%m-%d"),
            "shift_days": int(shift_days),
            "spearman_eigenvector_rank": _safe_float(sp),
            "kendall_eigenvector_rank": _safe_float(kt),
        }

        for k in topk_values:
            top_b = _topk(cb_c, int(k))
            top_f = _topk(cf_c, int(k))
            out[f"top{k}_jaccard_nodes"] = _safe_float(_jaccard(top_b, top_f))

        eb = _topm_edges(_pyg_to_edge_table(gb, bubble_firms), int(topm_edges))
        ef = _topm_edges(_pyg_to_edge_table(gf, frm_firms), int(topm_edges))
        set_b = set(zip(eb["u"], eb["v"]))
        set_f = set(zip(ef["u"], ef["v"]))
        out["top_edges_jaccard"] = _safe_float(_jaccard(list(set_b), list(set_f)))

        inter = sorted(set_b & set_f)
        if len(inter) >= 2:
            wb = []
            wf = []
            eb_map = {(u, v): float(w) for u, v, w in eb[["u", "v", "weight"]].itertuples(index=False)}
            ef_map = {(u, v): float(w) for u, v, w in ef[["u", "v", "weight"]].itertuples(index=False)}
            for u, v in inter:
                wb.append(float(abs(eb_map[(u, v)])))
                wf.append(float(abs(ef_map[(u, v)])))
            wb_arr = np.asarray(wb, dtype=float)
            wf_arr = np.asarray(wf, dtype=float)
            if float(np.nanstd(wb_arr)) <= 0.0 or float(np.nanstd(wf_arr)) <= 0.0:
                out["top_edges_weight_corr"] = float("nan")
            else:
                try:
                    out["top_edges_weight_corr"] = float(np.corrcoef(wb_arr, wf_arr)[0, 1])
                except Exception:
                    out["top_edges_weight_corr"] = float("nan")
        else:
            out["top_edges_weight_corr"] = float("nan")

        rows.append(out)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date_bubble").reset_index(drop=True)
    return df, summary


def write_similarity_summary_table(
    *,
    timeseries_df: pd.DataFrame,
    out_tex: str,
    caption: str = "Bubble vs FRM similarity over time (aligned by nearest prior date).",
    label: str = "tab:bubble_vs_frm_similarity",
) -> None:
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    metrics = [
        "spearman_eigenvector_rank",
        "kendall_eigenvector_rank",
        "top5_jaccard_nodes",
        "top10_jaccard_nodes",
        "top_edges_jaccard",
        "top_edges_weight_corr",
    ]
    rows = []
    for m in metrics:
        if m not in timeseries_df.columns:
            continue
        s = pd.to_numeric(timeseries_df[m], errors="coerce")
        vals = s.to_numpy(dtype=float, copy=True)
        finite = np.isfinite(vals)
        if not finite.any():
            mean = std = vmin = vmax = float("nan")
        else:
            mean = float(np.nanmean(vals))
            std = float(np.nanstd(vals))
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        rows.append(
            {
                "metric": m,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
            }
        )
    summary = pd.DataFrame(rows)

    def fmt(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{float(x):.3f}"

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{l r r r r}\n\\hline\n")
        f.write("Metric & Mean & Std & Min & Max \\\\\n\\hline\n")
        for _, r in summary.iterrows():
            name = str(r["metric"]).replace("_", "\\_")
            f.write(f"{name} & {fmt(r['mean'])} & {fmt(r['std'])} & {fmt(r['min'])} & {fmt(r['max'])} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\end{table}\n")


def plot_similarity_timeseries(
    *,
    timeseries_df: pd.DataFrame,
    out_fig: str,
    metrics: Sequence[str] = ("spearman_eigenvector_rank", "top10_jaccard_nodes"),
) -> None:
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    if timeseries_df.empty:
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "No aligned dates for similarity computation.", ha="center", va="center")
        plt.tight_layout()
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close()
        return

    df = timeseries_df.copy()
    df["date"] = pd.to_datetime(df["date_bubble"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    # downsample for readability (monthly mean)
    df_m = df.resample("ME").mean(numeric_only=True)

    plt.figure(figsize=(10, 4))
    for m in metrics:
        if m not in df_m.columns:
            continue
        plt.plot(df_m.index, df_m[m], label=m.replace("_", " "))
    corr_like = any(("spearman" in m) or ("kendall" in m) or ("corr" in m) for m in metrics)
    plt.ylim((-1, 1) if corr_like else (0, 1))
    plt.title("Bubble vs FRM similarity over time (monthly mean)")
    plt.xlabel("Date")
    plt.ylabel("Similarity")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
