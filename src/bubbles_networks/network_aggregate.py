import os
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def build_aggregate_overlap_graph(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    bubble_sheet="Breakdowns",
) -> nx.DiGraph:
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

    for i, row_i in bubble_data.iterrows():
        for j, row_j in bubble_data.iterrows():
            if i >= j:
                continue
            overlap_start = max(row_i["Start_Date"], row_j["Start_Date"])
            overlap_end = min(row_i["End_Date"], row_j["End_Date"])
            if pd.isna(overlap_start) or pd.isna(overlap_end):
                continue
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days <= 0:
                continue
            if row_i["Start_Date"] < row_j["Start_Date"]:
                G.add_edge(row_i["Firm"], row_j["Firm"], weight=float(overlap_days))
            else:
                G.add_edge(row_j["Firm"], row_i["Firm"], weight=float(overlap_days))

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

