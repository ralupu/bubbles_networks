import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from torch_geometric.data import TemporalData

import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import TGNMemory, GCNConv, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

import pickle

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.decomposition import PCA



# Load bubble data
bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
delta_covar_data = pd.read_excel("data_ro/ResultResults_ro_bet_covars.xlsx", sheet_name = 'Delta CoVaR (K=95%)')  # Contains delta CoVaR values

# Map numerical indices to actual dates
dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y")
date_mapping = {i+1: date for i, date in enumerate(dates_df["Date"])}
bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)

# print("🔍 Checking Date Mapping (Numerical Index → Actual Date)")
# print(date_mapping)  # Verify if the mapping is correct
#
# print("\n🔍 Checking Start_Date and End_Date in bubble_data:")
# print(bubble_data[["Firm", "Start", "Start_Date", "End", "End_Date"]].tail(10))  # Verify actual values


# Convert delta_covar_data index to datetime using the mapping
delta_covar_data.index = pd.to_datetime(delta_covar_data['Date'], format='%d/%m/%Y', errors="coerce")
delta_covar_data = delta_covar_data.drop('Date', axis=1)

# Now correctly extract time steps
time_windows = delta_covar_data.index  # Now contains real dates

# Create list of graphs per time step
temporal_graphs = []
for t, time_window in enumerate(time_windows[:-2]):
    time_window = pd.to_datetime(time_window)  # Ensure it's datetime

    # Filter firms that are active in this time window
    active_bubbles = bubble_data[
        (bubble_data["Start_Date"] <= time_window) & (bubble_data["End_Date"] >= time_window)
        ]

    # Create a directed graph for this time step
    G_t = nx.DiGraph()

    # Add firms as nodes
    for firm in active_bubbles["Firm"].unique():
        G_t.add_node(firm)

    # Add edges based on bubble overlaps in this time window
    for i, row_i in active_bubbles.iterrows():
        for j, row_j in active_bubbles.iterrows():
            if i < j:
                # Compute overlap
                overlap_start = max(row_i["Start_Date"], row_j["Start_Date"])
                overlap_end = min(row_i["End_Date"], row_j["End_Date"])
                overlap_days = (overlap_end - overlap_start).days

                if overlap_days > 0:
                    # Direction: Firm that entered first → Firm that entered later
                    if row_i["Start_Date"] < row_j["Start_Date"]:
                        G_t.add_edge(row_i["Firm"], row_j["Firm"], weight=overlap_days)
                    else:
                        G_t.add_edge(row_j["Firm"], row_i["Firm"], weight=overlap_days)

    # Attach delta CoVaR time-series data as node features
    missing_count = 0
    for firm in G_t.nodes():
        if firm in delta_covar_data.columns:
            try:
                G_t.nodes[firm]["delta_covar"] = float(delta_covar_data.loc[time_window, firm])
            except KeyError:
                print(f"Warning: No delta CoVaR data for {firm} at {time_window}. Assigning 0.")
                G_t.nodes[firm]["delta_covar"] = 0.0  # Assign default value
                missing_count += 1
        else:
            print(f"Warning: Firm {firm} not found in delta CoVaR data. Assigning 0.")
            G_t.nodes[firm]["delta_covar"] = 0.0  # Assign default value
            missing_count += 1

    # Debugging: Check if any node has a delta_covar attribute
    node_attrs = list(G_t.nodes(data=True))
    print(f"🔹 Time Step {t} - Firms with delta CoVaR: {len(G_t.nodes()) - missing_count}/{len(G_t.nodes())}")
    if len(node_attrs) > 0:
        print(f"Example Node Data: {node_attrs[0]}")  # Print one node's attributes

    # **Ensure at least one node has delta_covar before converting to PyG**
    if len(G_t.nodes()) > missing_count:
        # Convert to PyG format
        data_t = from_networkx(G_t, group_node_attrs=["delta_covar"])

        # Ensure tensor format
        data_t.x = torch.tensor(data_t.x, dtype=torch.float)

        temporal_graphs.append((t, data_t))
    else:
        print(f"⚠️ No valid delta CoVaR data for any firm at time step {t}, skipping this snapshot.")

print(f"✅ Created {len(temporal_graphs)} time-step graphs with delta CoVaR as features!")

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import TGNMemory, GCNConv, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

class TemporalGNN(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, memory_dim=64, num_nodes=1000):
        super(TemporalGNN, self).__init__()

        # Corrected IdentityMessage initialization
        self.message_module = IdentityMessage(raw_msg_dim=in_features, memory_dim=memory_dim, time_dim=1)

        # TGN Memory: Stores time-dependent node embeddings
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            memory_dim=memory_dim,
            message_module=self.message_module,
            aggregator_module=LastAggregator(),
            raw_msg_dim=in_features,
            time_dim = 1
        )

        # Graph Neural Network layers
        self.conv1 = GCNConv(memory_dim, hidden_dim)
        self.conv2 = TransformerConv(hidden_dim, out_features, heads=1)
        self.relu = nn.ReLU()

    def forward(self, data, time_step):
        x, edge_index = data.x, data.edge_index

        # Debugging: Print x shape before processing
        # print(f"Processing time step {time_step}: x shape = {x.shape}")

        # Ensure `x` is a tensor and convert to long type
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long, device=edge_index.device)
        else:
            x = x.long()  # Convert to integer tensor

        # Ensure `x` is properly flattened
        x = x.view(-1)

        if x.numel() == 0:  # Skip if x is empty
            print(f"⚠️ Skipping time step {time_step}: No nodes in graph")
            return None

        # Retrieve node memories (FIXED)
        memory, _ = self.memory(x)  # ✅ Extract only memory tensor, discard last_update

        # Pass through GNN layers
        x = self.conv1(memory, edge_index)  # ✅ Now passes a tensor, not a tuple
        x = self.relu(x)
        x = self.conv2(x, edge_index)

        return x


# Initialize model parameters
in_features = 1  # We are only using delta CoVaR as input
hidden_dim = 32
out_features = 1  # Predicting next delta CoVaR

# Initialize the model
model = TemporalGNN(in_features, hidden_dim, out_features)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Convert data to device
temporal_graphs = [(t, data.to(device)) for t, data in temporal_graphs]

# Train the model over time steps
num_epochs = 200
for epoch in range(num_epochs):
    print('epoch no {}'.format(epoch))
    model.train()
    optimizer.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True, device=device)  # FIXED: Initialize as tensor

    # Train on past time steps, ensuring `t + 1` is always valid
    for t in range(len(temporal_graphs) - 1):  # ✅ Stops at second-to-last index
        data_t = temporal_graphs[t][1]  # Get current time step data

        # Ensure `t + 1` exists before accessing `temporal_graphs`
        if t + 1 >= len(temporal_graphs):
            print(f"⚠️ Skipping time step {t}: No available target for t+1")
            continue  # Skip this iteration safely

        target = temporal_graphs[t + 1][1].x[:, 0]  # ✅ Now safe to access
        out = model(data_t, t).squeeze()

        # Debugging: Print shapes
        # print(f"🔍 Time step {t}: out shape {out.shape}, target shape {target.shape}")

        # Fix: Ensure `out` and `target` are 1D tensors
        out = out.view(-1)  # ✅ Ensure output is properly shaped
        target = target.view(-1)  # ✅ Ensure target is properly shaped

        # Fix: Ensure `out` and `target` have the same shape
        if out.shape[0] != target.shape[0]:
            # print(f"⚠️ Shape mismatch at time step {t}: out {out.shape}, target {target.shape}")
            min_size = min(out.shape[0], target.shape[0])
            out = out[:min_size]  # Truncate to match smaller size
            target = target[:min_size]  # Ensure same shape

        loss = loss + loss_fn(out, target)  # Compute loss safely

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


print("✅ Temporal GNN trained successfully!")

import torch
import torch.nn.functional as F

# Switch model to evaluation mode
model.eval()

# Initialize loss tracking
total_loss = 0
num_samples = 0

# Check if temporal_graphs has enough data
if len(temporal_graphs) > 1:
    with torch.no_grad():
        for t, data_t in temporal_graphs[:-3]:  # Use past time steps
            if t + 1 < len(temporal_graphs):  # Ensure next step exists
                out = model(data_t, t).squeeze()
                # Ensure `t + 1` is within bounds before accessing `temporal_graphs`
                if t + 1 >= len(temporal_graphs):
                    print(f"⚠️ Skipping time step {t}: No available target for t+1")
                    continue  # Skip this iteration safely

                target = temporal_graphs[t + 1][1].x[:, 0]  # ✅ Now safe to access

                # Ensure `out` and `target` exist before accessing `.shape[0]`
                if out is None or target is None:
                    print(f"⚠️ Skipping time step {t}: `out` or `target` is None")
                    continue  # ✅ Skip safely

                if not isinstance(out, torch.Tensor) or not isinstance(target, torch.Tensor):
                    print(f"⚠️ Skipping time step {t}: `out` or `target` is not a tensor")
                    continue  # ✅ Skip safely

                if out.numel() == 0 or target.numel() == 0:
                    print(f"⚠️ Skipping time step {t}: `out` or `target` is empty")
                    continue  # ✅ Skip safely

                # NEW CHECK: Ensure `out` and `target` have valid shapes before accessing `.shape[0]`
                if len(out.shape) == 0 or len(target.shape) == 0:
                    print(f"⚠️ Skipping time step {t}: `out` or `target` has an invalid shape")
                    continue  # ✅ Skip safely

                if out.shape[0] != target.shape[0]:
                    print(f"⚠️ Shape mismatch at time step {t}: out {out.shape}, target {target.shape}")
                    min_size = min(out.shape[0], target.shape[0])
                    out = out[:min_size]  # Truncate to match smaller size
                    target = target[:min_size]  # Ensure same shape

                loss = F.mse_loss(out, target)  # ✅ Now safe to compute loss

                total_loss += loss.item()
                num_samples += 1

    # Compute average loss safely
    if num_samples > 0:
        avg_mse = total_loss / num_samples
        print(f"📉 Model Evaluation - Average MSE: {avg_mse:.6f}")
    else:
        print("⚠️ No valid samples for evaluation. Check the `temporal_graphs` dataset.")
else:
    print("⚠️ Not enough time steps in `temporal_graphs`. Ensure the dataset is correctly loaded.")

if len(G_t.nodes()) > missing_count:
    temporal_graphs.append((t, data_t))  # Graph added only if nodes have features
else:
    print(f"⚠️ No valid delta CoVaR data for any firm at time step {t}, skipping this snapshot.")


# print(delta_covar_data.tail(10))  # Show last 10 rows
# # Debugging: Check which firms are active at each time step
# print(f"Processing time step {t} ({time_window}) - Firms Active: {len(active_bubbles)}")
# print(f"Firms in Graph at time {t}: {list(G_t.nodes())}")

# Save the variable to a pickle file
with open("temporal_graphs.pkl", "wb") as file:
    pickle.dump(temporal_graphs, file)
