"""Robustness experiments for the aggregate bubble-overlap network.

This module runs small sensitivity grids over overlap-network construction choices
and exports:
- per-configuration summaries under `results/robustness/` (ignored by git), and
- paper-facing tables/figures under `documents/` (tracked).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bubbles_networks.network_aggregate import OverlapNetworkConfig, build_aggregate_overlap_graph


@dataclass(frozen=True)
class RobustnessRun:
    """Single robustness configuration."""

    min_overlap_days: int
    edge_rule: str

    @property
    def config(self) -> OverlapNetworkConfig:
        return OverlapNetworkConfig(min_overlap_days=int(self.min_overlap_days), edge_rule=str(self.edge_rule))

    @property
    def run_id(self) -> str:
        return f"min{int(self.min_overlap_days)}__{str(self.edge_rule)}"


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / len(sa | sb))


def _graph_metrics(G) -> Dict[str, float]:
    n = int(G.number_of_nodes())
    m = int(G.number_of_edges())
    density = float(m / (n * (n - 1))) if n > 1 else 0.0
    w = np.array([float(G[u][v].get("weight", 1.0)) for u, v in G.edges()], dtype=float) if m else np.array([])
    out = {
        "nodes": float(n),
        "edges": float(m),
        "density": float(density),
        "w_min": float(np.min(w)) if w.size else 0.0,
        "w_median": float(np.median(w)) if w.size else 0.0,
        "w_mean": float(np.mean(w)) if w.size else 0.0,
        "w_max": float(np.max(w)) if w.size else 0.0,
    }
    try:
        import networkx as nx

        out["weak_components"] = float(nx.number_weakly_connected_components(G)) if n else 0.0
    except Exception:
        out["weak_components"] = 0.0
    return out


def _top_k(items: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
    return sorted(items.items(), key=lambda kv: (-float(kv[1]), str(kv[0])))[:k]


def _top_nodes_table(G, *, k: int = 10) -> pd.DataFrame:
    try:
        import networkx as nx

        pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() else {}
        und = G.to_undirected()
        bet = nx.betweenness_centrality(und) if und.number_of_nodes() else {}
    except Exception:
        pagerank, bet = {}, {}

    out_deg = {str(n): float(v) for n, v in dict(G.out_degree()).items()}
    out_wdeg = {str(n): float(v) for n, v in dict(G.out_degree(weight="weight")).items()}
    pr = {str(n): float(v) for n, v in pagerank.items()}
    bt = {str(n): float(v) for n, v in bet.items()}

    rows = []
    for metric, values in [
        ("out_degree", out_deg),
        ("out_weighted_degree", out_wdeg),
        ("pagerank", pr),
        ("betweenness_undirected", bt),
    ]:
        for rank, (node, val) in enumerate(_top_k(values, k=k), start=1):
            rows.append({"metric": metric, "rank": int(rank), "node": str(node), "value": float(val)})
    return pd.DataFrame(rows)


def _write_robustness_tex(df: pd.DataFrame, out_tex: str) -> None:
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        f.write("\\begin{tabular}{r l r r r r}\n\\hline\n")
        f.write("Min overlap & Edge rule & Nodes & Edges & Density & Top10 Jaccard \\\\\n\\hline\n")
        for _, r in df.iterrows():
            edge_rule_tex = str(r["edge_rule"]).replace("_", "\\_")
            f.write(
                f"{int(r['min_overlap_days'])} & {edge_rule_tex} & "
                f"{int(r['nodes'])} & {int(r['edges'])} & {float(r['density']):.4f} & {float(r['top10_jaccard']):.2f} \\\\\n"
            )
        f.write("\\hline\n\\end{tabular}\n")
        f.write("\\caption{Robustness summary for overlap-network construction choices.}\n")
        f.write("\\label{tab:robustness_summary}\n")
        f.write("\\end{table}\n")


def _write_heatmap(df: pd.DataFrame, out_fig: str) -> None:
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    pivot = df.pivot(index="min_overlap_days", columns="edge_rule", values="top10_jaccard")
    pivot = pivot.reindex(index=[0, 5, 10], columns=["start_lead", "overlap_undirected_then_direct"])

    plt.figure(figsize=(7, 3.5))
    try:
        import seaborn as sns

        sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0.0, vmax=1.0, cmap="viridis")
    except Exception:
        plt.imshow(pivot.values, aspect="auto", vmin=0.0, vmax=1.0)
        plt.colorbar(label="Top-10 Jaccard")
        plt.xticks(range(len(pivot.columns)), list(pivot.columns), rotation=20, ha="right")
        plt.yticks(range(len(pivot.index)), list(pivot.index))
    plt.title("Top-10 stability vs baseline (Jaccard)")
    plt.xlabel("Edge rule")
    plt.ylabel("Min overlap days")
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()


def run_overlap_network_robustness(
    *,
    dataset: str = "unknown",
    bubble_file: str,
    date_sheet: str,
    bubble_sheet: str,
    runs: Sequence[RobustnessRun],
    results_dir: str = "results/robustness",
    out_summary_csv: str = "results/robustness/robustness_summary.csv",
    out_paper_tex: str = "documents/tables/table_robustness_summary.tex",
    out_paper_fig: str = "documents/figures/fig_robustness_heatmap.png",
    top_k: int = 10,
) -> pd.DataFrame:
    """Run a small robustness grid for the aggregate overlap network.

    Writes per-run `network_summary.csv` and `top10_nodes.csv` under `results_dir/<run_id>/`.
    Also writes an aggregate CSV summary, and paper-facing LaTeX/figure outputs.
    """

    os.makedirs(results_dir, exist_ok=True)

    runs_sorted = sorted(runs, key=lambda r: (int(r.min_overlap_days), str(r.edge_rule)))
    baseline = runs_sorted[0] if runs_sorted else None
    baseline_top10: List[str] = []

    summary_rows = []
    for run in runs_sorted:
        G = build_aggregate_overlap_graph(
            bubble_file=bubble_file,
            date_sheet=date_sheet,
            bubble_sheet=bubble_sheet,
            config=run.config,
        )

        metrics = _graph_metrics(G)
        top_df = _top_nodes_table(G, k=top_k)
        top_pr = top_df[top_df["metric"] == "pagerank"].sort_values(["rank", "node"])
        top10 = top_pr["node"].head(top_k).tolist()
        if not top10:
            top_w = top_df[top_df["metric"] == "out_weighted_degree"].sort_values(["rank", "node"])
            top10 = top_w["node"].head(top_k).tolist()

        if baseline and run.run_id == baseline.run_id:
            baseline_top10 = list(top10)

        j = _jaccard(baseline_top10, top10) if baseline else 1.0

        out_dir = os.path.join(results_dir, run.run_id)
        os.makedirs(out_dir, exist_ok=True)

        per_run_summary = {
            "dataset": str(dataset),
            "min_overlap_days": int(run.min_overlap_days),
            "edge_rule": str(run.edge_rule),
            **metrics,
            "top10_jaccard": float(j),
        }
        pd.DataFrame([per_run_summary]).to_csv(os.path.join(out_dir, "network_summary.csv"), index=False)
        top_df.to_csv(os.path.join(out_dir, "top10_nodes.csv"), index=False)

        summary_rows.append(per_run_summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["min_overlap_days", "edge_rule"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_summary_csv) or ".", exist_ok=True)
    summary_df.to_csv(out_summary_csv, index=False)

    _write_robustness_tex(summary_df, out_paper_tex)
    _write_heatmap(summary_df, out_paper_fig)

    return summary_df
