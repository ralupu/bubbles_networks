import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
import networkx as nx
import torch_geometric.utils as pyg_utils
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric_temporal.nn.recurrent import GConvGRU, A3TGCN, EvolveGCNO
import argparse


# === Argparse CLI ===
def parse_args():
    parser = argparse.ArgumentParser(description="TGNN forecasting pipeline")
    parser.add_argument("--hidden-size", type=int, default=32, help="TGNN hidden layer size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--model", type=str, default="gconvgru", choices=["gconvgru", "a3tgcn", "evolvegcn"],
                        help="TGNN variant")
    parser.add_argument("--lookback", type=int, default=1, help="Number of timesteps for temporal input (A3TGCN)")
    return parser.parse_args()


args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("temporal_graphs.pkl", "rb") as f:
    temporal_graphs = pickle.load(f)
temporal_graphs.sort(key=lambda x: x[0])

edge_indices = []
edge_weights = []
x_inputs = []
y_targets = []

for date, data in temporal_graphs:
    G_nx = pyg_utils.to_networkx(data, to_undirected=True)
    edges = list(G_nx.edges)
    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = None
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(edge_index.shape[1], dtype=torch.float)
    edge_indices.append(edge_index)
    edge_weights.append(edge_attr)

    features = []
    for _, attr in G_nx.nodes(data=True):
        delta_covar = attr.get("delta_covar", 0.0)
        bubble_duration = attr.get("bubble_duration", 0.0)
        features.append([delta_covar, bubble_duration])
    features = np.array(features)
    x_inputs.append(features)

    if len(G_nx) > 0 and len(edges) > 0:
        eigenvector_centrality = nx.eigenvector_centrality(G_nx, max_iter=1000)
        sorted_nodes = list(G_nx.nodes)
        eigenvector_values = np.array([eigenvector_centrality.get(node, 0.0) for node in sorted_nodes])
    else:
        eigenvector_values = np.zeros(len(G_nx))
    y_targets.append(eigenvector_values)

y_targets_shifted = y_targets[1:] + [y_targets[-1]]

# For a3tgcn: build temporal windows [num_nodes, num_features, lookback]
if args.model == "a3tgcn" and args.lookback > 1:
    padded_x = [x_inputs[0]] * (args.lookback - 1) + x_inputs  # pad start with first available
    x_windowed = []
    for i in range(args.lookback - 1, len(x_inputs)):
        win = [padded_x[j] for j in range(i - args.lookback + 1, i + 1)]
        win_stack = np.stack(win, axis=-1)  # [num_nodes, num_features, lookback]
        x_windowed.append(win_stack)
    x_inputs = x_windowed
    # Make sure y_targets_shifted is aligned
    y_targets_shifted = y_targets_shifted[args.lookback - 1:]
    edge_indices = edge_indices[args.lookback - 1:]
    edge_weights = edge_weights[args.lookback - 1:]

data = DynamicGraphTemporalSignal(
    edge_indices=edge_indices,
    edge_weights=edge_weights,
    features=x_inputs,
    targets=y_targets_shifted
)

snapshots = list(data)
split = int(len(snapshots) * 0.8)
train_snapshots = snapshots[:split]
test_snapshots = snapshots[split:]


class TGNNWrapper(nn.Module):
    def __init__(self, model_name, node_features, hidden_size, output_size, lookback=1):
        super().__init__()
        if model_name == "gconvgru":
            self.recurrent = GConvGRU(node_features, hidden_size, 1)
        elif model_name == "a3tgcn":
            self.recurrent = A3TGCN(node_features, hidden_size, lookback)
        elif model_name == "evolvegcn":
            self.recurrent = EvolveGCNO(node_features, hidden_size, 1)
        else:
            raise ValueError("Unknown model variant")
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index, edge_weight, h):
        h = self.recurrent(x, edge_index, edge_weight, h)
        out = self.linear(h)
        return out, h


lookback = args.lookback if args.model == "a3tgcn" else 1
model = TGNNWrapper(
    model_name=args.model,
    node_features=2,
    hidden_size=args.hidden_size,
    output_size=1,
    lookback=lookback
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
epochs = args.epochs

model.train()
for epoch in range(epochs):
    loss_epoch = 0
    h = None
    for snapshot in train_snapshots:
        if snapshot.edge_index.numel() == 0:
            continue

        if args.model == "a3tgcn":
            x = torch.tensor(snapshot.x, dtype=torch.float).to(device)  # [num_nodes, num_features, lookback]
        else:
            x = torch.tensor(snapshot.x, dtype=torch.float).to(device)  # [num_nodes, num_features]

        edge_index = snapshot.edge_index.to(device)
        edge_weight = snapshot.edge_weight.to(device) if snapshot.edge_weight is not None else None
        y = torch.tensor(snapshot.y, dtype=torch.float).unsqueeze(-1).to(device)

        y_hat, h = model(x, edge_index, edge_weight, h)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch += loss.item()
        if h is not None:
            h = h.detach()
    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss_epoch:.4f}")

model.eval()
all_y_true = []
all_y_pred = []
h = None

for snapshot in test_snapshots:
    if snapshot.edge_index.numel() == 0:
        continue

    if args.model == "a3tgcn":
        x = torch.tensor(snapshot.x, dtype=torch.float).to(device)
    else:
        x = torch.tensor(snapshot.x, dtype=torch.float).to(device)
    edge_index = snapshot.edge_index.to(device)
    edge_weight = snapshot.edge_weight.to(device) if snapshot.edge_weight is not None else None
    y = torch.tensor(snapshot.y, dtype=torch.float).unsqueeze(-1).to(device)
    with torch.no_grad():
        y_hat, h = model(x, edge_index, edge_weight, h)
    all_y_true.append(y.cpu().numpy().flatten())
    all_y_pred.append(y_hat.cpu().numpy().flatten())

all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)
mse = np.mean((all_y_true - all_y_pred) ** 2)
mae = np.mean(np.abs(all_y_true - all_y_pred))

print("\n✅ Evaluation Results:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")

plt.figure(figsize=(10, 5))
plt.scatter(all_y_true, all_y_pred, alpha=0.5)
plt.xlabel("True Eigenvector Centrality")
plt.ylabel("Predicted")
plt.title(
    f"TGNN Forecast Performance ({args.model}, hidden={args.hidden_size}, epochs={args.epochs}, lookback={lookback})")
plt.grid()
filename = f"figures/tgnn_forecast_performance_{args.model}_hidden{args.hidden_size}_epochs{args.epochs}_lookback{lookback}.png"
plt.savefig(filename, dpi=300)
plt.close()
print(f"✅ Saved forecast chart as {filename}")

print("✅ Forecasting pipeline completed.")


# run code:
# python tgnn_forecasting_module.py --hidden-size 128 --epochs 100 --model a3tgcn --lookback 5