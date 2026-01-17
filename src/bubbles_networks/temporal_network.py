import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import from_networkx


def build_temporal_graphs(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    covar_file="data/ro/ResultResults_ro_bet_covars.xlsx",
    bubble_sheet="Breakdowns",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    covar_sheet="Delta CoVaR (K=95%)",
    out_file="results/temporal_graphs.pkl",
    node_feature_keys=("delta_covar", "bubble_duration", "bubble_eigenvector"),
    min_edges=0,
    compute_centrality=True,
):
    """
    Build time-indexed bubble-overlap networks and save as a list of (date, PyG Data) tuples.

    - Node features: delta CoVaR (normalized), bubble duration (normalized), optional eigenvector centrality.
    - Edge direction: earlier bubble start -> later bubble start (weight=1).
    """
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    bubble_data = pd.read_excel(bubble_file, sheet_name=bubble_sheet)
    delta_covar_data = pd.read_excel(covar_file, sheet_name=covar_sheet)
    dates_df = pd.read_excel(bubble_file, sheet_name=date_sheet)

    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y", errors="coerce")
    date_mapping = {i + 1: date for i, date in enumerate(dates_df["Date"])}
    bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
    bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)
    delta_covar_data["Date"] = pd.to_datetime(delta_covar_data["Date"], format="%d/%m/%Y", errors="coerce")

    firms = bubble_data["Firm"].unique()
    temporal_graphs = []
    time_series = [t for t in delta_covar_data["Date"].unique() if pd.notna(t)]

    for t in time_series:
        G = nx.DiGraph()
        active_bubbles = bubble_data[(bubble_data["Start_Date"] <= t) & (bubble_data["End_Date"] >= t)]

        if t in delta_covar_data["Date"].values:
            delta_covar_at_t = delta_covar_data.set_index("Date").T[t].to_dict()
        else:
            delta_covar_at_t = {firm: 0.0 for firm in firms}

        bubble_duration_days = {firm: 0 for firm in firms}
        for firm in firms:
            bubble_info = active_bubbles[active_bubbles["Firm"] == firm]
            if not bubble_info.empty:
                delta = bubble_info["End_Date"].values[0] - bubble_info["Start_Date"].values[0]
                bubble_duration_days[firm] = int(delta.astype("timedelta64[D]").astype(int))

        scaler = MinMaxScaler()
        delta_vals = np.array(list(delta_covar_at_t.values()), dtype=float).reshape(-1, 1)
        dur_vals = np.array(list(bubble_duration_days.values()), dtype=float).reshape(-1, 1)
        delta_norm = scaler.fit_transform(delta_vals).flatten()
        dur_norm = scaler.fit_transform(dur_vals).flatten()

        for i, firm in enumerate(firms):
            G.add_node(firm, delta_covar=float(delta_norm[i]), bubble_duration=float(dur_norm[i]))

        for i, firm_i in enumerate(firms):
            for j, firm_j in enumerate(firms):
                if i == j:
                    continue
                bubble_i = active_bubbles[active_bubbles["Firm"] == firm_i]
                bubble_j = active_bubbles[active_bubbles["Firm"] == firm_j]
                if bubble_i.empty or bubble_j.empty:
                    continue
                if bubble_i["Start_Date"].values[0] < bubble_j["Start_Date"].values[0]:
                    G.add_edge(firm_i, firm_j, weight=1.0)

        if compute_centrality and G.number_of_edges() > 0:
            try:
                eigen = (
                    nx.eigenvector_centrality_numpy(G)
                    if G.number_of_nodes() > 1
                    else {n: 0.0 for n in G.nodes()}
                )
            except Exception:
                eigen = {n: 0.0 for n in G.nodes()}
            for n in G.nodes():
                G.nodes[n]["bubble_eigenvector"] = float(eigen.get(n, 0.0))
        elif compute_centrality:
            for n in G.nodes():
                G.nodes[n]["bubble_eigenvector"] = 0.0

        if G.number_of_edges() >= int(min_edges):
            pyg_graph = from_networkx(G, group_node_attrs=list(node_feature_keys))
            temporal_graphs.append((pd.to_datetime(t), pyg_graph))

    with open(out_file, "wb") as f:
        pickle.dump(temporal_graphs, f)

    print(f"Temporal graphs saved: {out_file} ({len(temporal_graphs)} snapshots)")


if __name__ == "__main__":
    build_temporal_graphs()
