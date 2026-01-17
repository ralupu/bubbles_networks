import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import QuantileRegressor
from torch_geometric.utils import from_networkx


def prices_to_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to log-returns.

    Assumes a "Date" column plus one column per asset.
    """
    prices_only = prices_df.drop(columns=["Date"])
    returns = np.log(prices_only).diff().fillna(0)
    returns.insert(0, "Date", prices_df["Date"])
    return returns


def build_dynamic_frm_graphs(
    returns_file="data_ro/ResultResults_ro_bet_returns.xlsx",
    prices_file=None,
    frm_window=250,
    start_date=None,
    quantile=0.95,
    min_edges=1,
    alpha=1.0,
    edge_threshold=1e-3,
    max_zero_frac=0.2,
    save_as="frm_graphs.pkl",
    centrality_features=True,
    parallel=True,
    n_jobs=-1,
):
    """
    Build dynamic FRM networks using rolling-window quantile regression.

    Each time step t builds a directed graph where edges represent estimated tail-dependence
    (via quantile regression coefficients) over a rolling window.
    """
    if os.path.exists(returns_file):
        df = pd.read_excel(returns_file)
        print(f"Loaded returns from {returns_file}")
    elif prices_file is not None and os.path.exists(prices_file):
        prices = pd.read_excel(prices_file)
        df = prices_to_returns(prices)
        print(f"Converted prices to returns from {prices_file}")
    else:
        raise FileNotFoundError("Neither returns_file nor prices_file found.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(start_date)]

    def build_graph_for_day(t: int):
        window_df = df.iloc[t - frm_window : t].copy()
        date_t = df["Date"].iloc[t]

        max_allowed_zeros = int(max_zero_frac * frm_window)
        assets_to_keep = [
            col
            for col in window_df.columns
            if col != "Date" and (window_df[col] == 0).sum() < max_allowed_zeros
        ]
        if len(assets_to_keep) < 2:
            return None

        window_df = window_df[["Date"] + assets_to_keep]
        available_firms = [col for col in window_df.columns if col != "Date"]

        G = nx.DiGraph()
        for firm in available_firms:
            G.add_node(firm)

        for target in available_firms:
            y = window_df[target].values
            X_firms = [f for f in available_firms if f != target]
            X = window_df[X_firms].values
            if np.isnan(y).any() or np.isnan(X).any():
                continue

            try:
                qr = QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")
                qr.fit(X, y)
                coefs = qr.coef_
            except Exception:
                continue

            for i, other in enumerate(X_firms):
                weight = float(abs(coefs[i]))
                if weight > edge_threshold:
                    G.add_edge(other, target, weight=weight)

        if centrality_features and G.number_of_edges() > 0:
            try:
                eigen = (
                    nx.eigenvector_centrality_numpy(G)
                    if G.number_of_nodes() > 1
                    else {n: 0.0 for n in G.nodes()}
                )
            except Exception:
                eigen = {n: 0.0 for n in G.nodes()}

            degree = dict(G.degree(weight=None))
            betweenness = (
                nx.betweenness_centrality(G) if G.number_of_nodes() > 1 else {n: 0.0 for n in G.nodes()}
            )

            for n in G.nodes():
                G.nodes[n]["frm_eigenvector"] = float(eigen.get(n, 0.0))
                G.nodes[n]["frm_degree"] = float(degree.get(n, 0.0))
                G.nodes[n]["frm_betweenness"] = float(betweenness.get(n, 0.0))

        if G.number_of_edges() >= int(min_edges):
            pyg_graph = from_networkx(G, group_node_attrs=["frm_eigenvector", "frm_degree", "frm_betweenness"])
            return (date_t, pyg_graph)

        return None

    iterator = range(frm_window, len(df))
    if parallel:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(build_graph_for_day)(t) for t in iterator)
    else:
        results = [build_graph_for_day(t) for t in iterator]

    frm_graphs = [g for g in results if g is not None]
    print(f"Number of FRM graphs created: {len(frm_graphs)} / {len(df) - frm_window}")

    with open(save_as, "wb") as f:
        pickle.dump(frm_graphs, f)
    print(f"FRM graphs saved: {save_as} ({len(frm_graphs)} snapshots)")


if __name__ == "__main__":
    build_dynamic_frm_graphs()
