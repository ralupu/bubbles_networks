import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


@dataclass(frozen=True)
class OverlapNetworkConfig:
    """Configuration for aggregate bubble-overlap network construction."""

    min_overlap_days: int = 0
    edge_rule: str = "start_lead"  # {"start_lead", "overlap_undirected_then_direct"}


def build_aggregate_overlap_graph(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    bubble_sheet="Breakdowns",
    config: Optional[OverlapNetworkConfig] = None,
) -> nx.DiGraph:
    """Build an aggregate directed overlap network from bubble episode timing.

    Edge rules:
    - "start_lead": earlier bubble start -> later bubble start (per-overlap instance).
    - "overlap_undirected_then_direct": aggregate overlaps per firm pair, then direct the edge
      by the average lead/lag of bubble start dates across overlapping episodes.

    Parameters
    - bubble_file/date_sheet/bubble_sheet: Excel inputs.
    - config: OverlapNetworkConfig controlling overlap threshold and edge direction rule.
    """

    cfg = config or OverlapNetworkConfig()
    if cfg.edge_rule not in {"start_lead", "overlap_undirected_then_direct"}:
        raise ValueError(f"Unknown edge_rule: {cfg.edge_rule}")
    if int(cfg.min_overlap_days) < 0:
        raise ValueError(f"min_overlap_days must be >= 0, got {cfg.min_overlap_days}")

    dates_df = pd.read_excel(bubble_file, sheet_name=date_sheet)
    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y", errors="coerce")
    date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

    bubble_data = pd.read_excel(bubble_file, sheet_name=bubble_sheet)
    bubble_data = bubble_data.sort_values(by=["Firm", "Start"]).copy()
    bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
    bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)

    G = nx.DiGraph()
    for firm in sorted(bubble_data["Firm"].unique()):
        G.add_node(firm)

    min_days = int(cfg.min_overlap_days)
    overlap_records = []
    rows = list(bubble_data.itertuples(index=False))
    for i in range(len(rows)):
        ri = rows[i]
        for j in range(i + 1, len(rows)):
            rj = rows[j]
            si, ei = ri.Start_Date, ri.End_Date
            sj, ej = rj.Start_Date, rj.End_Date
            if pd.isna(si) or pd.isna(ei) or pd.isna(sj) or pd.isna(ej):
                continue
            overlap_start = max(si, sj)
            overlap_end = min(ei, ej)
            if pd.isna(overlap_start) or pd.isna(overlap_end):
                continue
            overlap_days = int((overlap_end - overlap_start).days)
            if overlap_days <= 0:
                continue
            if min_days > 0 and overlap_days < min_days:
                continue
            overlap_records.append((ri.Firm, rj.Firm, overlap_days, si, sj))

    if cfg.edge_rule == "start_lead":
        for fi, fj, overlap_days, si, sj in overlap_records:
            if si < sj:
                u, v = fi, fj
            elif sj < si:
                u, v = fj, fi
            else:
                u, v = (fi, fj) if str(fi) <= str(fj) else (fj, fi)
            prev = float(G[u][v]["weight"]) if G.has_edge(u, v) else 0.0
            G.add_edge(u, v, weight=prev + float(overlap_days))
        return G

    # overlap_undirected_then_direct: aggregate by unordered firm pair, direct by mean start lag.
    pair_overlap_sum = {}
    pair_lags = {}
    for fi, fj, overlap_days, si, sj in overlap_records:
        a, b = (fi, fj) if str(fi) <= str(fj) else (fj, fi)
        lag_days = int((sj - si).days) if (str(a) == str(fi) and str(b) == str(fj)) else int((si - sj).days)
        key = (a, b)
        pair_overlap_sum[key] = float(pair_overlap_sum.get(key, 0.0)) + float(overlap_days)
        pair_lags.setdefault(key, []).append(float(lag_days))

    for (a, b), w in sorted(pair_overlap_sum.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        lags = pair_lags.get((a, b), [])
        mean_lag = float(sum(lags) / len(lags)) if lags else 0.0
        if mean_lag > 0:
            u, v = a, b
        elif mean_lag < 0:
            u, v = b, a
        else:
            u, v = a, b
        prev = float(G[u][v]["weight"]) if G.has_edge(u, v) else 0.0
        G.add_edge(u, v, weight=prev + float(w))

    return G


def plot_aggregate_network_circular(G: nx.DiGraph, out_path: str) -> Optional[str]:
    if G.number_of_edges() == 0:
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pagerank = nx.pagerank(G, alpha=0.85)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=[w / 100 for w in edge_weights])
    node_size = [5000 * pagerank[firm] for firm in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=node_size, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def run_aggregate_network_analysis(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    bubble_sheet="Breakdowns",
    out_path="figures/bubble_network_circular.png",
):
    G = build_aggregate_overlap_graph(
        bubble_file=bubble_file,
        date_sheet=date_sheet,
        bubble_sheet=bubble_sheet,
    )

    out = plot_aggregate_network_circular(G, out_path)
    if out is None:
        print("No edges found; skipping aggregate network plot.")
        return
    print(f"Aggregate network analysis plot saved: {out}")
