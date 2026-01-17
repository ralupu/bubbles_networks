import os
import pickle

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch_geometric


def compute_centrality_measures(temporal_graphs, company_mapping):
    all_centrality = []

    for time_step, G_t in temporal_graphs:
        if G_t is None or getattr(G_t, "num_nodes", 0) == 0:
            continue

        G_nx = torch_geometric.utils.to_networkx(G_t, to_undirected=True)
        degree_centrality = nx.degree_centrality(G_nx)
        betweenness_centrality = nx.betweenness_centrality(G_nx)
        eigenvector_centrality = nx.eigenvector_centrality(G_nx, max_iter=1000)

        for node in G_nx.nodes:
            company_name = company_mapping.get(node, f"Unknown_{node}")
            all_centrality.append(
                {
                    "Date": time_step,
                    "Company": company_name,
                    "Degree": degree_centrality.get(node, 0.0),
                    "Betweenness": betweenness_centrality.get(node, 0.0),
                    "Eigenvector": eigenvector_centrality.get(node, 0.0),
                }
            )

    return pd.DataFrame(all_centrality)


def run_centrality_analysis(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    bubble_sheet="Breakdowns",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    temporal_graphs_pkl="results/temporal_graphs.pkl",
    out_dynamics="figures/centrality_dynamics.png",
    out_heatmap="figures/centrality_heatmap.png",
):
    os.makedirs(os.path.dirname(out_dynamics) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_heatmap) or ".", exist_ok=True)

    dates_df = pd.read_excel(bubble_file, sheet_name=date_sheet)
    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y", errors="coerce")

    bubble_data = pd.read_excel(bubble_file, sheet_name=bubble_sheet)
    company_mapping = {i: firm for i, firm in enumerate(bubble_data["Firm"].unique())}

    with open(temporal_graphs_pkl, "rb") as f:
        temporal_graphs = pickle.load(f)

    centrality_df = compute_centrality_measures(temporal_graphs, company_mapping)
    if centrality_df.empty:
        print("No centrality data produced; skipping plots.")
        return

    top_companies = centrality_df.groupby("Company")["Eigenvector"].mean().nlargest(5).index
    filtered_df = centrality_df[centrality_df["Company"].isin(top_companies)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for i, measure in enumerate(["Degree", "Betweenness", "Eigenvector"]):
        sns.scatterplot(data=filtered_df, x="Date", y=measure, hue="Company", style="Company", ax=axes[i], s=50)
        axes[i].set_title(f"{measure} Centrality Over Time")
    plt.tight_layout()
    plt.savefig(out_dynamics, dpi=300, bbox_inches="tight")
    plt.close()

    centrality_df = centrality_df.groupby(["Company", "Date"], as_index=False).agg({"Eigenvector": "mean"})
    heatmap_data = centrality_df.pivot(index="Company", columns="Date", values="Eigenvector")
    if heatmap_data.empty:
        print("No data available for heatmap; skipping heatmap generation.")
        return

    plt.figure(figsize=(12, 8))
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns)
    heatmap_data.columns = heatmap_data.columns.strftime("%Y-%m-%d")

    sns.heatmap(heatmap_data, cmap="RdBu", center=0, cbar_kws={"label": "Eigenvector Centrality"})
    plt.ylabel("Company")
    plt.xlabel("")
    plt.yticks(rotation=0)

    xticks = np.linspace(0, len(heatmap_data.columns) - 1, 10).astype(int)
    plt.xticks(ticks=xticks, labels=[heatmap_data.columns[i] for i in xticks], rotation=45)

    plt.tight_layout()
    plt.savefig(out_heatmap, dpi=300, bbox_inches="tight")
    plt.close()

    print("Centrality analysis completed and plots saved.")


if __name__ == "__main__":
    run_centrality_analysis()
