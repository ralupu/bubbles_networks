import pandas as pd
import matplotlib.pyplot as plt


def generate_bubble_overlap_chart():
    # Load data
    dates_df = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='BUB (CVM= WB, CVQ=95%, L=0)')
    dates_df["Date"] = pd.to_datetime(dates_df['Date'], format='%d/%m/%Y')
    date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

    bubble_data = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
    bubble_data = bubble_data.sort_values(by=["Firm", "Start"])

    firms = bubble_data["Firm"].unique()
    firm_positions = {firm: i for i, firm in enumerate(firms)}

    plt.figure(figsize=(12, 6))

    boom_label_added = False
    burst_label_added = False

    for _, row in bubble_data.iterrows():
        y = firm_positions[row["Firm"]]
        start_date = date_mapping.get(row["Start"], None)
        peak_date = date_mapping.get(row["Peak"], None)
        end_date = date_mapping.get(row["End"], None)

        if start_date and peak_date and end_date:
            plt.hlines(
                y, start_date, peak_date, colors="green", linewidth=2,
                label="Boom Phase" if not boom_label_added else None
            )
            boom_label_added = True

            plt.hlines(
                y, peak_date, end_date, colors="red", linewidth=2,
                label="Burst Phase" if not burst_label_added else None
            )
            burst_label_added = True

    plt.yticks(range(len(firms)), firms)
    plt.xticks(rotation=45)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.savefig('figures/overlapping_bubbles.png', dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Overlapping bubbles chart saved.")
