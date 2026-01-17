import argparse
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric_temporal.nn.recurrent import A3TGCN, EvolveGCNO, GConvGRU
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal


@dataclass(frozen=True)
class GraphSnapshots:
    dates: List[np.datetime64]
    graphs: List[Data]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TGNN forecasting pipeline")
    parser.add_argument("--hidden-size", type=int, default=32, help="TGNN hidden layer size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--model",
        type=str,
        default="gconvgru",
        choices=["gconvgru", "a3tgcn", "evolvegcn"],
        help="TGNN variant",
    )
    parser.add_argument("--lookback", type=int, default=1, help="Number of timesteps for temporal input (A3TGCN)")
    parser.add_argument(
        "--mode",
        type=str,
        default="bubble",
        choices=["bubble", "frm", "both"],
        help="Which graph/features to use",
    )
    parser.add_argument("--bubble-file", type=str, default="temporal_graphs.pkl", help="Bubble graph pkl file")
    parser.add_argument("--frm-file", type=str, default="frm_graphs.pkl", help="FRM graph pkl file")
    parser.add_argument(
        "--edge-source",
        type=str,
        default="bubble",
        choices=["bubble", "frm"],
        help="(mode=both) Which graph provides edges",
    )
    return parser.parse_args(argv)


def _load_pickle(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_graph_snapshots(pkl_path: str) -> GraphSnapshots:
    raw = _load_pickle(pkl_path)
    if not isinstance(raw, list):
        raise TypeError(f"Expected a list in {pkl_path}, got {type(raw)}")

    dates: List[np.datetime64] = []
    graphs: List[Data] = []
    for item in raw:
        if not (isinstance(item, tuple) and len(item) == 2):
            raise TypeError(f"Expected (date, Data) tuples in {pkl_path}; got: {type(item)}")
        date, graph = item
        if not isinstance(graph, Data):
            raise TypeError(f"Expected torch_geometric.data.Data in {pkl_path}; got {type(graph)}")
        dates.append(np.datetime64(date))
        graphs.append(graph)

    order = np.argsort(np.array(dates))
    dates_sorted = [dates[i] for i in order]
    graphs_sorted = [graphs[i] for i in order]
    return GraphSnapshots(dates=dates_sorted, graphs=graphs_sorted)


def align_by_date(a: GraphSnapshots, b: GraphSnapshots) -> List[Tuple[np.datetime64, Data, Data]]:
    a_by_date = {d: g for d, g in zip(a.dates, a.graphs)}
    b_by_date = {d: g for d, g in zip(b.dates, b.graphs)}
    common = sorted(set(a_by_date) & set(b_by_date))
    return [(d, a_by_date[d], b_by_date[d]) for d in common]


def get_edge_weight_tensor(g: Data) -> Optional[torch.Tensor]:
    if hasattr(g, "edge_weight") and g.edge_weight is not None:
        return g.edge_weight
    if hasattr(g, "weight") and g.weight is not None:
        return g.weight
    return None


def eigenvector_centrality_targets(g: Data) -> np.ndarray:
    num_nodes = int(getattr(g, "num_nodes", 0) or 0)
    if num_nodes == 0:
        return np.zeros((0,), dtype=float)
    if g.edge_index is None or g.edge_index.numel() == 0:
        return np.zeros((num_nodes,), dtype=float)

    G_nx = pyg_utils.to_networkx(g, to_undirected=True)
    if len(G_nx) <= 1 or G_nx.number_of_edges() == 0:
        return np.zeros((num_nodes,), dtype=float)

    try:
        eigen = nx.eigenvector_centrality(G_nx, max_iter=1000)
    except Exception:
        return np.zeros((num_nodes,), dtype=float)

    return np.array([float(eigen.get(i, 0.0)) for i in range(num_nodes)], dtype=float)


def build_temporal_signal_from_snapshots(
    graphs: List[Data],
    lookback: int,
    use_a3tgcn_windowing: bool,
) -> Tuple[DynamicGraphTemporalSignal, int]:
    edge_indices: List[torch.Tensor] = []
    edge_weights: List[torch.Tensor] = []
    x_inputs: List[np.ndarray] = []
    y_targets: List[np.ndarray] = []

    for g in graphs:
        x = getattr(g, "x", None)
        if x is None:
            raise ValueError("Graph snapshot missing `x` node features.")
        x_inputs.append(x.detach().cpu().numpy())

        edge_index = g.edge_index if g.edge_index is not None else torch.empty((2, 0), dtype=torch.long)
        edge_indices.append(edge_index)

        w = get_edge_weight_tensor(g)
        num_edges = int(edge_index.shape[1])
        if w is None:
            edge_weights.append(torch.ones((num_edges,), dtype=torch.float))
        else:
            edge_weights.append(w.detach().to(dtype=torch.float))
        y_targets.append(eigenvector_centrality_targets(g))

    if not y_targets:
        raise ValueError("No snapshots available after loading graphs.")

    y_targets_shifted = y_targets[1:] + [y_targets[-1]]

    if use_a3tgcn_windowing and lookback > 1:
        padded_x = [x_inputs[0]] * (lookback - 1) + x_inputs
        x_windowed: List[np.ndarray] = []
        for i in range(lookback - 1, len(x_inputs)):
            win = [padded_x[j] for j in range(i - lookback + 1, i + 1)]
            x_windowed.append(np.stack(win, axis=-1))  # [num_nodes, num_features, lookback]
        x_inputs = x_windowed
        y_targets_shifted = y_targets_shifted[lookback - 1 :]
        edge_indices = edge_indices[lookback - 1 :]
        edge_weights = edge_weights[lookback - 1 :]

    node_feature_dim = int(x_inputs[0].shape[1]) if x_inputs else 0
    data = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=x_inputs,
        targets=y_targets_shifted,
    )
    return data, node_feature_dim


class TGNNWrapper(nn.Module):
    def __init__(self, model_name: str, node_features: int, hidden_size: int, output_size: int, lookback: int = 1):
        super().__init__()
        if model_name == "gconvgru":
            self.recurrent = GConvGRU(node_features, hidden_size, 1)
        elif model_name == "a3tgcn":
            self.recurrent = A3TGCN(node_features, hidden_size, lookback)
        elif model_name == "evolvegcn":
            self.recurrent = EvolveGCNO(node_features, hidden_size, 1)
        else:
            raise ValueError(f"Unknown model variant: {model_name}")
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor], h: Optional[torch.Tensor]
    ):
        h = self.recurrent(x, edge_index, edge_weight, h)
        out = self.linear(h)
        return out, h


def run(args: argparse.Namespace) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "bubble":
        bubble = load_graph_snapshots(args.bubble_file)
        graphs = bubble.graphs
    elif args.mode == "frm":
        frm = load_graph_snapshots(args.frm_file)
        graphs = frm.graphs
    elif args.mode == "both":
        bubble = load_graph_snapshots(args.bubble_file)
        frm = load_graph_snapshots(args.frm_file)
        aligned = align_by_date(bubble, frm)
        if not aligned:
            raise ValueError("No common dates found between bubble and FRM graph pickles.")
        graphs = []
        for _, gb, gf in aligned:
            if gb.num_nodes != gf.num_nodes:
                raise ValueError(
                    f"Node count mismatch for aligned date: bubble={gb.num_nodes}, frm={gf.num_nodes}. "
                    "Dual-mode requires consistent node indexing."
                )
            x = torch.cat([gb.x, gf.x], dim=1)
            g_edges = gb if args.edge_source == "bubble" else gf
            g_combined = Data(x=x, edge_index=g_edges.edge_index)
            w = get_edge_weight_tensor(g_edges)
            if w is not None:
                g_combined.weight = w
            graphs.append(g_combined)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    lookback = int(args.lookback) if args.model == "a3tgcn" else 1
    temporal_signal, node_feature_dim = build_temporal_signal_from_snapshots(
        graphs=graphs, lookback=lookback, use_a3tgcn_windowing=(args.model == "a3tgcn")
    )

    snapshots = list(temporal_signal)
    if len(snapshots) < 5:
        raise ValueError(f"Not enough snapshots to train/test (got {len(snapshots)}).")
    split = int(len(snapshots) * 0.8)
    train_snapshots = snapshots[:split]
    test_snapshots = snapshots[split:]

    model = TGNNWrapper(
        model_name=args.model,
        node_features=node_feature_dim,
        hidden_size=int(args.hidden_size),
        output_size=1,
        lookback=lookback,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(int(args.epochs)):
        loss_epoch = 0.0
        h = None
        for snapshot in train_snapshots:
            if snapshot.edge_index is None or snapshot.edge_index.numel() == 0:
                continue

            x = snapshot.x
            x = x if torch.is_tensor(x) else torch.as_tensor(x)
            x = x.to(device=device, dtype=torch.float)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = snapshot.edge_weight.to(device) if snapshot.edge_weight is not None else None
            y = snapshot.y
            y = y if torch.is_tensor(y) else torch.as_tensor(y)
            y = y.to(device=device, dtype=torch.float).unsqueeze(-1)

            y_hat, h = model(x, edge_index, edge_weight, h)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch += float(loss.item())
            if h is not None:
                h = h.detach()
        print(f"Epoch {epoch + 1}/{int(args.epochs)} | Train Loss: {loss_epoch:.4f}")

    model.eval()
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []
    h = None

    for snapshot in test_snapshots:
        if snapshot.edge_index is None or snapshot.edge_index.numel() == 0:
            continue
        x = snapshot.x
        x = x if torch.is_tensor(x) else torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float)
        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_weight.to(device) if snapshot.edge_weight is not None else None
        y = snapshot.y
        y = y if torch.is_tensor(y) else torch.as_tensor(y)
        y = y.to(device=device, dtype=torch.float).unsqueeze(-1)
        with torch.no_grad():
            y_hat, h = model(x, edge_index, edge_weight, h)
        all_y_true.append(y.cpu().numpy().flatten())
        all_y_pred.append(y_hat.cpu().numpy().flatten())

    if not all_y_true:
        raise ValueError("No valid test snapshots (all had empty edge_index).")

    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    print("\nEvaluation Results:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")

    os.makedirs("figures", exist_ok=True)
    filename = (
        f"figures/tgnn_forecast_performance_{args.mode}_{args.model}_hidden{args.hidden_size}_epochs{args.epochs}_"
        f"lookback{lookback}.png"
    )
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Eigenvector Centrality (next step)")
    plt.ylabel("Predicted")
    plt.title(f"TGNN Forecast Performance ({args.mode}, {args.model})")
    plt.grid()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved forecast chart: {filename}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
