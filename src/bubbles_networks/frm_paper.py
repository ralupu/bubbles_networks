"""FRM paper-facing outputs (diagnostics + comparisons + sensitivity)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from bubbles_networks.frm_network import FRMConfig, aggregate_frm_snapshots, build_frm_snapshot_sequence, load_returns_for_frm
from bubbles_networks.network_aggregate import build_aggregate_overlap_graph
from bubbles_networks.validation import write_network_diagnostics


def _top10_by_pagerank(G: nx.DiGraph, k: int = 10) -> List[str]:
    if G.number_of_nodes() == 0:
        return []
    pr = nx.pagerank(G, alpha=0.85) if G.number_of_edges() else {n: 0.0 for n in G.nodes()}
    items = sorted(pr.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))
    return [str(n) for n, _ in items[:k]]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def write_frm_vs_bubble_comparison(
    *,
    bubble_graph: nx.DiGraph,
    frm_graph: nx.DiGraph,
    out_fig: str,
) -> Dict[str, float]:
    """Compare aggregate FRM vs bubble-overlap graphs and export a simple figure."""

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    top_b = _top10_by_pagerank(bubble_graph, k=10)
    top_f = _top10_by_pagerank(frm_graph, k=10)
    top10_jaccard = _jaccard(top_b, top_f)

    pr_b = nx.pagerank(bubble_graph, alpha=0.85) if bubble_graph.number_of_nodes() else {}
    pr_f = nx.pagerank(frm_graph, alpha=0.85) if frm_graph.number_of_nodes() else {}
    nodes = sorted(set(pr_b.keys()) | set(pr_f.keys()), key=lambda x: str(x))
    s_b = pd.Series({n: float(pr_b.get(n, 0.0)) for n in nodes})
    s_f = pd.Series({n: float(pr_f.get(n, 0.0)) for n in nodes})
    spearman = float(s_b.corr(s_f, method="spearman")) if len(nodes) >= 2 else 0.0

    plt.figure(figsize=(6.5, 3.5))
    vals = [top10_jaccard, spearman]
    labels = ["Top-10 Jaccard\n(Pagerank)", "Spearman\n(Pagerank)"]
    plt.bar(range(len(vals)), vals, color=["#4C72B0", "#55A868"])
    plt.ylim(0, 1)
    plt.xticks(range(len(vals)), labels)
    for i, v in enumerate(vals):
        plt.text(i, min(1.0, v + 0.04), f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    plt.title("Bubble vs FRM network similarity (aggregate)")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()

    return {"top10_jaccard_pagerank": float(top10_jaccard), "spearman_pagerank": float(spearman)}


def run_frm_small_and_export_paper_outputs(
    *,
    cfg: FRMConfig,
    bubble_file: str,
    date_sheet: str,
    bubble_sheet: str,
    out_summary_csv: str,
    out_table_tex: str,
    out_degree_fig: str,
    out_compare_fig: str,
) -> Dict[str, str]:
    """Run FRM snapshots and export paper-facing diagnostics + comparison."""

    _, firms = load_returns_for_frm(cfg)
    firm_mapping = {i: f for i, f in enumerate(firms)}

    frm_graphs = build_frm_snapshot_sequence(cfg)
    frm_agg = aggregate_frm_snapshots(frm_graphs, firm_mapping=firm_mapping)

    write_network_diagnostics(
        aggregate_graph=frm_agg,
        temporal_graphs_pkl=os.path.join(cfg.out_dir, "frm_graphs.pkl"),
        firm_mapping=firm_mapping,
        out_csv=out_summary_csv,
        out_tex=out_table_tex,
        out_degree_fig=out_degree_fig,
        tex_caption="FRM network diagnostics (aggregate and temporal average).",
        tex_label="tab:frm_network_summary",
        tex_aggregate_name="FRM (agg)",
        tex_temporal_name="FRM (avg)",
    )

    bubble_agg = build_aggregate_overlap_graph(bubble_file=bubble_file, date_sheet=date_sheet, bubble_sheet=bubble_sheet)
    sim = write_frm_vs_bubble_comparison(bubble_graph=bubble_agg, frm_graph=frm_agg, out_fig=out_compare_fig)

    return {
        "frm_snapshots": str(len(frm_graphs)),
        "frm_nodes": str(int(frm_agg.number_of_nodes())),
        "frm_edges": str(int(frm_agg.number_of_edges())),
        "frm_top10_jaccard_pagerank": f"{sim['top10_jaccard_pagerank']:.4f}",
        "frm_spearman_pagerank": f"{sim['spearman_pagerank']:.4f}",
    }


def run_frm_sensitivity_grid(
    *,
    base_cfg: FRMConfig,
    window_sizes: Sequence[int],
    out_tex: str,
    out_fig: str,
    top_k: int = 10,
) -> None:
    """Run a tiny FRM sensitivity grid (window sizes only) and export to paper."""

    rows = []
    baseline_top10: List[str] = []

    for w in sorted(set(int(x) for x in window_sizes)):
        cfg = FRMConfig(**{**base_cfg.__dict__, "window_size": int(w), "out_dir": os.path.join(base_cfg.out_dir, f"sensitivity_w{w}")})
        _, firms = load_returns_for_frm(cfg)
        firm_mapping = {i: f for i, f in enumerate(firms)}
        graphs = build_frm_snapshot_sequence(cfg)
        agg = aggregate_frm_snapshots(graphs, firm_mapping=firm_mapping)

        pr_top10 = _top10_by_pagerank(agg, k=top_k)
        if not baseline_top10:
            baseline_top10 = list(pr_top10)
        j = _jaccard(baseline_top10, pr_top10)
        n = int(agg.number_of_nodes())
        m = int(agg.number_of_edges())
        density = float(m / (n * (n - 1))) if n > 1 else 0.0
        rows.append(
            {
                "window_size": int(w),
                "snapshots": int(len(graphs)),
                "nodes": int(n),
                "edges": int(m),
                "density": float(density),
                "top10_jaccard": float(j),
            }
        )

    df = pd.DataFrame(rows).sort_values("window_size").reset_index(drop=True)

    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{r r r r r}\n\\hline\n")
        f.write("Window & Snapshots & Edges & Density & Top10 Jaccard \\\\\n\\hline\n")
        for _, r in df.iterrows():
            f.write(
                f"{int(r['window_size'])} & {int(r['snapshots'])} & {int(r['edges'])} & {float(r['density']):.4f} & {float(r['top10_jaccard']):.2f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{FRM sensitivity (window size) with top-K per node thresholding.}\n")
        f.write("\\label{tab:frm_sensitivity}\n")
        f.write("\\end{table}\n")

    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    plt.figure(figsize=(6, 2.8))
    x = df["window_size"].tolist()
    y = df["top10_jaccard"].tolist()
    plt.imshow(np.array([y]), aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    plt.yticks([0], ["Top-10\nJaccard"])
    plt.xticks(range(len(x)), [str(xx) for xx in x])
    for i, val in enumerate(y):
        plt.text(i, 0, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black")
    plt.xlabel("Window size (days)")
    plt.title("FRM sensitivity (top-10 stability vs baseline)")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()

