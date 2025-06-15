import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def run_aggregate_network_analysis():
    # Load data
    dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
    dates_df["Date"] = pd.to_datetime(dates_df['Date'], format='%d/%m/%Y')
    date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

    bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
    bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
    bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)

    # Build network
    G = nx.DiGraph()
    for firm in bubble_data["Firm"].unique():
        G.add_node(firm)

    for i, row_i in bubble_data.iterrows():
        for j, row_j in bubble_data.iterrows():
            if i < j:
                overlap_start = max(row_i["Start_Date"], row_j["Start_Date"])
                overlap_end = min(row_i["End_Date"], row_j["End_Date"])
                overlap_days = (overlap_end - overlap_start).days
                if overlap_days > 0:
                    if row_i["Start_Date"] < row_j["Start_Date"]:
                        G.add_edge(row_i["Firm"], row_j["Firm"], weight=overlap_days)
                    else:
                        G.add_edge(row_j["Firm"], row_i["Firm"], weight=overlap_days)

    if G.number_of_edges() == 0:
        print("No edges found!")
        return

    pagerank = nx.pagerank(G, alpha=0.85)
    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 10))
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=[w / 100 for w in edge_weights])
    node_size = [5000 * pagerank[firm] for firm in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=node_size, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.savefig("figures/bubble_network_circular.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Aggregate network analysis completed and plot saved.")
