import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class FRMConfig:
    """Configuration for FRM (quantile-regression) dynamic network construction."""

    returns_file: str = "data/ro/ResultResults_ro_bet_returns.xlsx"
    prices_file: Optional[str] = None

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    max_days: Optional[int] = None

    window_size: int = 250
    step_size: int = 1
    quantile: float = 0.05
    alpha: float = 0.0

    max_zero_frac: float = 0.2
    max_firms: Optional[int] = None  # keeps first N firm columns

    threshold_method: str = "top_k_in"  # {"top_k_in", "abs_beta"}
    top_k: int = 5
    abs_beta_tau: float = 1e-3
    weight_mode: str = "abs"  # {"abs", "signed"}

    min_edges: int = 1
    parallel: bool = False
    n_jobs: int = -1

    out_dir: str = "results/frm"


def load_returns_for_frm(cfg: FRMConfig) -> Tuple[pd.DataFrame, List[str]]:
    """Load and sanitize a Date-indexed returns dataframe for FRM."""

    if os.path.exists(cfg.returns_file):
        df = pd.read_excel(cfg.returns_file)
        source = cfg.returns_file
    elif cfg.prices_file is not None and os.path.exists(cfg.prices_file):
        prices = pd.read_excel(cfg.prices_file)
        df = prices_to_returns(prices)
        source = cfg.prices_file
    else:
        raise FileNotFoundError("Neither returns_file nor prices_file found.")

    if "Date" not in df.columns:
        raise ValueError(f"FRM requires a Date column in returns data: {source}")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    if cfg.start_date is not None:
        df = df[df["Date"] >= pd.to_datetime(cfg.start_date)]
    if cfg.end_date is not None:
        df = df[df["Date"] <= pd.to_datetime(cfg.end_date)]
    if cfg.max_days is not None:
        df = df.head(int(cfg.max_days))

    firm_cols = [c for c in df.columns if c != "Date"]
    if cfg.max_firms is not None:
        firm_cols = firm_cols[: int(cfg.max_firms)]

    df = df[["Date"] + firm_cols].copy()
    return df, firm_cols


def build_frm_snapshot_sequence(cfg: FRMConfig) -> List[Tuple[pd.Timestamp, object]]:
    """Build dynamic FRM networks as a list of (date, PyG Data) snapshots.

    Formal definition (per snapshot):
    - For each target firm j, fit a quantile regression at quantile q:
        r_j(t) = a_j + sum_{i != j} beta_{i->j}(t) * r_i(t) + eps
      using a rolling window of length `window_size` ending at t.
    - Add directed edge i -> j if it passes the threshold rule:
        * top_k_in: keep the top-K incoming edges (largest |beta|) per target j
        * abs_beta: keep edges with |beta| > tau
    - Edge weight is either |beta| ("abs") or beta ("signed").
    """

    if int(cfg.window_size) <= 1:
        raise ValueError("window_size must be > 1")
    if int(cfg.step_size) <= 0:
        raise ValueError("step_size must be > 0")
    if cfg.threshold_method not in {"top_k_in", "abs_beta"}:
        raise ValueError(f"Unknown threshold_method: {cfg.threshold_method}")
    if cfg.weight_mode not in {"abs", "signed"}:
        raise ValueError(f"Unknown weight_mode: {cfg.weight_mode}")

    df, firms = load_returns_for_frm(cfg)
    if len(firms) < 2:
        return []

    window = int(cfg.window_size)
    step = int(cfg.step_size)

    max_allowed_zeros = int(cfg.max_zero_frac * window)

    def build_graph_for_index(t: int):
        window_df = df.iloc[t - window : t].copy()
        date_t = df["Date"].iloc[t]

        eligible = [c for c in firms if (window_df[c] == 0).sum() < max_allowed_zeros]
        if len(eligible) < 2:
            # keep node universe consistent; just emit empty graph for this date
            G = nx.DiGraph()
            for firm in firms:
                G.add_node(firm)
            pyg = from_networkx(G)
            return (pd.to_datetime(date_t), pyg)

        window_df = window_df[eligible]
        G = nx.DiGraph()
        for firm in firms:
            G.add_node(firm)

        for target in eligible:
            y = window_df[target].to_numpy(dtype=float)
            X_firms = [f for f in eligible if f != target]
            X = window_df[X_firms].to_numpy(dtype=float)
            if np.isnan(y).any() or np.isnan(X).any():
                continue

            try:
                qr = QuantileRegressor(quantile=float(cfg.quantile), alpha=float(cfg.alpha), solver="highs")
                qr.fit(X, y)
                coefs = np.asarray(qr.coef_, dtype=float)
            except Exception:
                continue

            candidates = []
            for i, other in enumerate(X_firms):
                beta = float(coefs[i])
                w = float(abs(beta)) if cfg.weight_mode == "abs" else float(beta)
                candidates.append((other, target, beta, w))

            if cfg.threshold_method == "top_k_in":
                # keep top-K incoming edges per target (by |beta|); stable tie-break by firm name
                candidates.sort(key=lambda x: (-abs(float(x[2])), str(x[0])))
                candidates = candidates[: int(cfg.top_k)]
            else:
                candidates = [c for c in candidates if abs(float(c[2])) > float(cfg.abs_beta_tau)]

            for src, dst, beta, w in candidates:
                if abs(float(w)) <= 0.0:
                    continue
                G.add_edge(src, dst, weight=float(w), beta=float(beta))

        if G.number_of_edges() < int(cfg.min_edges):
            return None

        pyg = from_networkx(G)
        return (pd.to_datetime(date_t), pyg)

    iterator = list(range(window, len(df), step))
    if cfg.parallel:
        results = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(build_graph_for_index)(t) for t in iterator)
    else:
        results = [build_graph_for_index(t) for t in iterator]

    graphs = [g for g in results if g is not None]
    graphs.sort(key=lambda x: x[0])

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "frm_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)
    with open(os.path.join(cfg.out_dir, "frm_firms.json"), "w", encoding="utf-8") as f:
        json.dump(list(firms), f, indent=2)

    with open(os.path.join(cfg.out_dir, "frm_graphs.pkl"), "wb") as f:
        pickle.dump(graphs, f)

    print(f"[frm] Built snapshots={len(graphs)} window={cfg.window_size} step={cfg.step_size} firms={len(firms)}")
    return graphs


def aggregate_frm_snapshots(
    frm_graphs: Sequence[Tuple[pd.Timestamp, object]], *, firm_mapping: Dict[int, str]
) -> nx.DiGraph:
    """Aggregate FRM snapshots into a single directed graph by summing edge weights.

    Parameters
    - frm_graphs: list of (date, PyG Data)
    - firm_mapping: node-index -> firm name mapping (stable across snapshots)
    """

    G = nx.DiGraph()
    for idx, name in sorted(firm_mapping.items(), key=lambda kv: int(kv[0])):
        G.add_node(str(name))

    for _, pyg in frm_graphs:
        edge_index = getattr(pyg, "edge_index", None)
        if edge_index is None:
            continue
        w = getattr(pyg, "weight", None)
        if w is None:
            w = getattr(pyg, "edge_weight", None)
        edges = edge_index.detach().cpu().numpy().T.astype(int) if hasattr(edge_index, "detach") else np.asarray(edge_index).T
        weights = w.detach().cpu().numpy().astype(float) if hasattr(w, "detach") else (np.asarray(w).astype(float) if w is not None else None)

        for k, (u, v) in enumerate(edges):
            uu = str(firm_mapping.get(int(u), f"Unknown_{u}"))
            vv = str(firm_mapping.get(int(v), f"Unknown_{v}"))
            ww = float(weights[k]) if weights is not None and k < len(weights) else 1.0
            if G.has_edge(uu, vv):
                G[uu][vv]["weight"] = float(G[uu][vv].get("weight", 0.0)) + ww
            else:
                G.add_edge(uu, vv, weight=ww)
    return G


def build_dynamic_frm_graphs(
    returns_file="data/ro/ResultResults_ro_bet_returns.xlsx",
    prices_file=None,
    frm_window=250,
    start_date=None,
    quantile=0.95,
    min_edges=1,
    alpha=1.0,
    edge_threshold=1e-3,
    max_zero_frac=0.2,
    save_as="results/frm_graphs.pkl",
    centrality_features=True,
    parallel=True,
    n_jobs=-1,
):
    """
    Build dynamic FRM networks using rolling-window quantile regression.

    Each time step t builds a directed graph where edges represent estimated tail-dependence
    (via quantile regression coefficients) over a rolling window.
    """
    os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)

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
