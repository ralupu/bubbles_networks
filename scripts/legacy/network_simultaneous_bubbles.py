import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load bubble data (Replace 'bubbles.csv' with your actual file)
dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
dates_df["Date"] = pd.to_datetime(dates_df['Date'], format='%d/%m/%Y')
# Generate a mapping from index position to actual date
date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

bubble_data  = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')

# Map numerical Start and End indices to actual dates
bubble_data["Start_Date"] = bubble_data["Start"].map(date_mapping)
bubble_data["End_Date"] = bubble_data["End"].map(date_mapping)

# Create a directed graph
G = nx.DiGraph()

# Add nodes (firms)
for firm in bubble_data["Firm"].unique():
    G.add_node(firm)

# Fix the overlap detection condition using actual dates
for i, row_i in bubble_data.iterrows():
    for j, row_j in bubble_data.iterrows():
        if i < j:
            # Compute actual intersection period
            overlap_start = max(row_i["Start_Date"], row_j["Start_Date"])
            overlap_end = min(row_i["End_Date"], row_j["End_Date"])
            overlap_days = (overlap_end - overlap_start).days

            # Ensure there is a **strictly positive overlap**
            if overlap_days > 0:
                # Direction: Firm that entered first → Firm that entered later
                if row_i["Start_Date"] < row_j["Start_Date"]:
                    G.add_edge(row_i["Firm"], row_j["Firm"], weight=overlap_days)
                else:
                    G.add_edge(row_j["Firm"], row_i["Firm"], weight=overlap_days)

# Check if graph has edges
if G.number_of_edges() == 0:
    print("No edges found! Check data and overlap conditions.")
else:
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# Compute PageRank for systemic importance
pagerank = nx.pagerank(G, alpha=0.85)

# 📌 Use Circular Layout for better visualization
pos = nx.circular_layout(G)

# Set figure size
plt.figure(figsize=(10, 10))

# Draw edges with width based on weight
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color="gray", width=[w / 100 for w in edge_weights])

# Draw nodes (scaled by PageRank)
node_size = [5000 * pagerank[firm] for firm in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_color="skyblue", node_size=node_size, alpha=0.7)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10)

# Set title
# plt.title("Bubble Synchronization Network: Circular Layout", fontsize=14)

# Save the figure as a PNG file
plt.savefig("figures/bubble_network_circular.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()

