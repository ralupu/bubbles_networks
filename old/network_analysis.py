import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch_geometric
import plotly.express as px
import plotly.io as pio


def compute_centrality_measures(temporal_graphs):
    """
    Compute Degree, Betweenness, and Eigenvector centrality over time.

    Args:
        temporal_graphs: List of tuples (time, PyTorch Geometric graph).

    Returns:
        centrality_df: DataFrame containing centrality measures for each company over time.
    """

    all_centrality = []

    # Iterate over all available time steps in the temporal graphs
    for t, (time_step, G_t) in enumerate(temporal_graphs):  # ✅ FIXED unpacking
        if G_t is None or G_t.num_nodes == 0:  # ✅ FIXED: Use num_nodes for PyTorch Geometric
            continue  # Skip empty networks

        # Convert PyTorch Geometric graph to NetworkX for centrality computation
        G_nx = torch_geometric.utils.to_networkx(G_t, to_undirected=True)

        # Compute centrality measures
        degree_centrality = nx.degree_centrality(G_nx)
        betweenness_centrality = nx.betweenness_centrality(G_nx)
        eigenvector_centrality = nx.eigenvector_centrality(G_nx, max_iter=1000)

        # Store results in a DataFrame format
        for node in G_nx.nodes:
            all_centrality.append({
                "Time": time_step,  # ✅ Use actual time step, not index
                "Company": node,
                "Degree": degree_centrality.get(node, 0),
                "Betweenness": betweenness_centrality.get(node, 0),
                "Eigenvector": eigenvector_centrality.get(node, 0),
            })

    # Convert to DataFrame
    centrality_df = pd.DataFrame(all_centrality)
    return centrality_df


import matplotlib.pyplot as plt
import seaborn as sns


def plot_centrality_dynamics(centrality_df, dates_df, bubble_data, top_n=5):
    """
    Plots the evolution of centrality measures over time for the top_n most central firms.

    Parameters:
    - centrality_df: DataFrame with ['Time', 'Company', 'Degree', 'Betweenness', 'Eigenvector']
    - dates_df: DataFrame with 'Date' column (pd.to_datetime)
    - bubble_data: DataFrame containing 'Firm' (company names) and corresponding indices
    - top_n: Number of top companies to display
    """
    # Extract last 1100 dates matching temporal_graphs
    last_1100_dates = dates_df['Date'].iloc[-1100:].reset_index(drop=True)

    # Create a mapping from numerical indices to real company names using bubble_data
    company_mapping = {idx: firm for idx, firm in enumerate(bubble_data['Firm'].unique())}

    # Replace numerical company indices in centrality_df with real names
    centrality_df["Company"] = centrality_df["Company"].map(company_mapping)

    # Select top companies (based on highest average Eigenvector centrality)
    top_companies = (
        centrality_df.groupby("Company")["Eigenvector"].mean().nlargest(top_n).index
    )

    # Filter to include only selected companies
    filtered_df = centrality_df[centrality_df["Company"].isin(top_companies)]

    # Create a mapping from time indices to actual dates
    time_to_date = dict(zip(filtered_df["Time"].unique(), last_1100_dates))

    # Replace time indices with actual dates
    filtered_df["Date"] = filtered_df["Time"].map(time_to_date)

    # Initialize figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Define centrality measures
    centrality_measures = ["Degree", "Betweenness", "Eigenvector"]
    y_labels = ["Degree", "Betweenness", "Eigenvector"]

    # Plot each centrality measure
    for i, (measure, ylabel) in enumerate(zip(centrality_measures, y_labels)):
        ax = axes[i]

        # Scatter plot only (NO BLACK EDGES & NO LINES)
        sns.scatterplot(
            data=filtered_df, x="Date", y=measure, hue="Company", style="Company", ax=ax, s=50, legend=(i == 2)
        )

        ax.set_ylabel(ylabel)
        ax.set_title(f"{measure} Centrality Over Time")

        # Only show x-axis labels on the last subplot
        if i < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Time (Date)")
            ax.tick_params(axis="x", rotation=45)

        # **Fix for missing legend removal**
    if axes[0].get_legend() is not None:
        axes[0].get_legend().remove()
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    # Extract the legend from the last plot and move it below the figure
    handles, labels = axes[2].get_legend_handles_labels()
    if axes[2].get_legend() is not None:
        axes[2].get_legend().remove()

    fig.legend(handles, labels, title="Company", loc="lower center", ncol=top_n, bbox_to_anchor=(0.5, -0.05))

    # Adjust layout
    plt.tight_layout()
    plt.savefig("figures/centrality_dynamics.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()


def plot_centrality_heatmap(centrality_df, dates_df, measure="Eigenvector", save_path="centrality_heatmap.png"):
    """
    Plots and saves a heatmap of the specified centrality measure over time using Plotly.

    Parameters:
    - centrality_df: DataFrame with ["Time", "Company", measure] columns.
    - dates_df: DataFrame with actual dates corresponding to each time step in column ['Date'].
    - measure: The centrality measure to visualize (default: "Eigenvector").
    - save_path: Path to save the output heatmap image (default: "centrality_heatmap.png").
    """

    # **Ensure Date is Datetime and Time is Integer**
    dates_df["Date"] = pd.to_datetime(dates_df["Date"])
    centrality_df["Time"] = centrality_df["Time"].astype(int)

    # **Map Time to Real Dates**
    time_mapping = dict(zip(dates_df.index, dates_df["Date"]))
    centrality_df["Date"] = centrality_df["Time"].map(time_mapping)

    # **Drop NaN Dates**
    centrality_df.dropna(subset=["Date"], inplace=True)

    # **Aggregate duplicates by taking the mean**
    centrality_df = centrality_df.groupby(["Company", "Date"], as_index=False).agg({measure: "mean"})

    # **Pivot for Heatmap**
    heatmap_data = centrality_df.pivot(index="Company", columns="Date", values=measure)

    # **Select Date Ticks Every 50 Time Steps (Optimized X-Axis)**
    tick_positions = heatmap_data.columns[::50]

    # **Create a Plotly Heatmap**
    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x="Date", y="Company", color=f"{measure} Centrality"),
        x=heatmap_data.columns,  # Dates on x-axis
        y=heatmap_data.index,  # Companies on y-axis
        color_continuous_scale="RdBu",  # Valid Plotly colorscale
    )

    # **Optimize Layout**
    fig.update_layout(
        title=f"{measure} Centrality Heatmap",
        xaxis=dict(
            tickangle=-45,
            tickmode="array",
            tickvals=tick_positions,  # Optimized downsampling of X-axis
            ticktext=[str(date.date()) for date in tick_positions],  # Show only the date
        ),
        yaxis=dict(tickmode="array"),
        coloraxis_colorbar=dict(title=f"{measure} Centrality"),
        template="plotly_white"  # White background
    )

    # **Show Interactive Plot**
    pio.renderers.default = "browser"  # Force plotly to open in browser
    fig.show()

    # **Save as PNG (Optimized)**
    try:
        image_data = fig.to_image(format="png", scale=2)
        with open(save_path, "wb") as f:
            f.write(image_data)

        print(f"✅ Heatmap saved successfully as {save_path}")
    except Exception as e:
        print(f"❌ Error saving heatmap: {e}")



dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y")

bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')

with open("temporal_graphs.pkl", "rb") as file:
    temporal_graphs = pickle.load(file)
# Compute centrality
centrality_df = compute_centrality_measures(temporal_graphs)

# Plot centrality dynamics
plot_centrality_dynamics(centrality_df, dates_df, bubble_data, top_n=5)
# Plot centrality heatmap
plot_centrality_heatmap(centrality_df, dates_df, measure="Eigenvector", save_path="figures/centrality_heatmap.png")
