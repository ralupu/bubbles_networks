import numpy as np
import pandas as pd
import networkx as nx
import pickle
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import MinMaxScaler


def build_temporal_graphs():
    # Load input data
    bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
    delta_covar_data = pd.read_excel("data_ro/ResultResults_ro_bet_covars.xlsx", sheet_name='Delta CoVaR (K=95%)')

    dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y")
    date_mapping = {i+1: date for i, date in enumerate(dates_df["Date"])}
    bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
    bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)
    delta_covar_data["Date"] = pd.to_datetime(delta_covar_data["Date"], format='%d/%m/%Y', errors="coerce")

    firms = bubble_data["Firm"].unique()
    temporal_graphs = []
    time_series = delta_covar_data["Date"].unique()

    for t in time_series:
        firms_at_t = delta_covar_data[delta_covar_data["Date"] == t].columns[1:]
        G = nx.DiGraph()

        active_bubbles = bubble_data[(bubble_data["Start_Date"] <= t) & (bubble_data["End_Date"] >= t)]
        if t in delta_covar_data["Date"].values:
            delta_covar_at_t = delta_covar_data.set_index("Date").T[t].to_dict()
        else:
            delta_covar_at_t = {firm: 0.0 for firm in firms}

        bubble_duration = {firm: 0 for firm in firms}
        for firm in firms:
            bubble_info = active_bubbles[active_bubbles["Firm"] == firm]
            if not bubble_info.empty:
                bubble_duration[firm] = (bubble_info["End_Date"].values[0] - bubble_info["Start_Date"].values[0]).astype("timedelta64[D]").astype(int)

        scaler = MinMaxScaler()
        firm_values = np.array(list(delta_covar_at_t.values())).reshape(-1, 1)
        normalized_values = scaler.fit_transform(firm_values).flatten()
        duration_values = np.array(list(bubble_duration.values())).reshape(-1, 1)
        normalized_durations = scaler.fit_transform(duration_values).flatten()

        for i, firm in enumerate(firms):
            G.add_node(firm, delta_covar=normalized_values[i], bubble_duration=normalized_durations[i])

        for i, firm_i in enumerate(firms):
            for j, firm_j in enumerate(firms):
                if i != j:
                    bubble_i = active_bubbles[active_bubbles["Firm"] == firm_i]
                    bubble_j = active_bubbles[active_bubbles["Firm"] == firm_j]
                    if not bubble_i.empty and not bubble_j.empty:
                        if bubble_i["Start_Date"].values[0] < bubble_j["Start_Date"].values[0]:
                            G.add_edge(firm_i, firm_j, weight=1)

        if len(G.nodes) > 0:
            temporal_graphs.append((pd.to_datetime(t), from_networkx(G, group_node_attrs=["delta_covar"])))

    with open("temporal_graphs.pkl", "wb") as f:
        pickle.dump(temporal_graphs, f)

    print("✅ Temporal graphs built and saved successfully.")
