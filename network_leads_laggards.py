import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import GCNConv, TransformerConv
from torch_geometric.utils import to_networkx
from sklearn.preprocessing import MinMaxScaler

# Load data
bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
delta_covar_data = pd.read_excel("data_ro/ResultResults_ro_bet_covars.xlsx", sheet_name = 'Delta CoVaR (K=95%)')

# Convert dates to datetime
dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y")
date_mapping = {i+1: date for i, date in enumerate(dates_df["Date"])}
bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)
delta_covar_data["Date"] = pd.to_datetime(delta_covar_data["Date"], format='%d/%m/%Y', errors="coerce")

# Ensure unique firms
firms = bubble_data["Firm"].unique()


# # Reshape delta_covar_data to have a 'Firm' column instead of firms as columns
# delta_covar_data_long = delta_covar_data.melt(id_vars=["Date"], var_name="Firm", value_name="delta_covar")


def construct_network(time_step, firm_list):
    """Constructs a network at a given time step with enriched node features."""
    G = nx.DiGraph()
    active_bubbles = bubble_data[(bubble_data["Start_Date"] <= time_step) & (bubble_data["End_Date"] >= time_step)]

    # Extract delta CoVaR values correctly
    if time_step in delta_covar_data["Date"].values:
        delta_covar_at_t = delta_covar_data.set_index("Date").T[time_step].to_dict()
    else:
        delta_covar_at_t = {firm: 0.0 for firm in firm_list}  # Assign zero if no data

    # Compute bubble duration for each firm
    bubble_duration = {firm: 0 for firm in firm_list}
    for firm in firm_list:
        bubble_info = active_bubbles[active_bubbles["Firm"] == firm]
        if not bubble_info.empty:
            bubble_duration[firm] = (bubble_info["End_Date"].values[0] - bubble_info["Start_Date"].values[0]).astype(
                "timedelta64[D]").astype(int)

    # Normalize delta CoVaR and bubble duration
    scaler = MinMaxScaler()
    firm_values = np.array(list(delta_covar_at_t.values())).reshape(-1, 1)
    normalized_values = scaler.fit_transform(firm_values).flatten()

    duration_values = np.array(list(bubble_duration.values())).reshape(-1, 1)
    normalized_durations = scaler.fit_transform(duration_values).flatten()

    # Add nodes with normalized features
    for i, firm in enumerate(firm_list):
        G.add_node(firm, delta_covar=normalized_values[i], bubble_duration=normalized_durations[i])

    # Add edges based on bubble start times
    for i, firm_i in enumerate(firm_list):
        for j, firm_j in enumerate(firm_list):
            if i != j:
                bubble_i = active_bubbles[active_bubbles["Firm"] == firm_i]
                bubble_j = active_bubbles[active_bubbles["Firm"] == firm_j]

                if not bubble_i.empty and not bubble_j.empty:
                    if bubble_i["Start_Date"].values[0] < bubble_j["Start_Date"].values[0]:
                        G.add_edge(firm_i, firm_j, weight=1)

    return G


# Check for isolated nodes
def check_isolated_nodes(graphs):
    """Identifies isolated nodes in temporal graphs."""
    for t, G_t in graphs:
        # Ensure we're working with a networkx graph
        if isinstance(G_t, nx.Graph) or isinstance(G_t, nx.DiGraph):
            isolated_nodes = list(nx.isolates(G_t))
        else:
            print(f"⚠️ Skipping time {t}: G_t is not a networkx graph")
            continue  # Skip if not networkx

        if len(isolated_nodes) > 0:
            print(f"Time {t}: Isolated nodes - {isolated_nodes}")


temporal_graphs = []
time_series = delta_covar_data["Date"].unique()

for t in time_series:
    firms_at_t = delta_covar_data[delta_covar_data["Date"] == t].columns[1:]  # Exclude "Date" column
    G_t = construct_network(t, firms_at_t)

    if len(G_t.nodes) > 0:
        temporal_graphs.append((t, from_networkx(G_t, group_node_attrs=["delta_covar"])))

# Save the constructed graphs
with open("temporal_graphs.pkl", "wb") as f:
    torch.save(temporal_graphs, f)

print("✅ Temporal network construction completed successfully with dynamic edge weights!")
