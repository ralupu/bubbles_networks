import pandas as pd
import matplotlib.pyplot as plt

# Load the data
dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
dates_df["Date"] = pd.to_datetime(dates_df['Date'], format='%d/%m/%Y')
# Generate a mapping from index position to actual date
date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

bubble_data  = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')  # Replace with the actual path

# Sort the data by firm and time start
bubble_data  = bubble_data.sort_values(by=["Firm", "Start"])

# Create a mapping for firms to numerical y-axis positions
firms = bubble_data ["Firm"].unique()
firm_positions = {firm: i for i, firm in enumerate(firms)}

# Set up the plot
plt.figure(figsize=(12, 6))

# Plot boom and burst phases
for _, row in bubble_data .iterrows():
    y = firm_positions[row["Firm"]]

    # Get the corresponding dates
    start_date = date_mapping.get(row["Start"], None)
    peak_date = date_mapping.get(row["Peak"], None)
    end_date = date_mapping.get(row["End"], None)

    if start_date and peak_date and end_date:
        plt.hlines(y, start_date, peak_date, colors="green", linewidth=2, label="Boom Phase" if y == 0 else "")
        plt.hlines(y, peak_date, end_date, colors="red", linewidth=2, label="Burst Phase" if y == 0 else "")

# Formatting the plot
plt.yticks(range(len(firms)), firms)
# plt.xlabel("Time")
# plt.ylabel("Firms")
# plt.title("Bubble Phases for Romanian Companies")
# plt.legend()

# Format x-axis with dates
plt.xticks(rotation=45)
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Show the plot
# plt.show()`
plt.savefig('figures/overlapping_bubbles.png')