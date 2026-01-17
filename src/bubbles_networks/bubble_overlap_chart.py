import os

import matplotlib.pyplot as plt
import pandas as pd


def generate_bubble_overlap_chart(
    bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx",
    date_sheet="BUB (CVM= WB, CVQ=95%, L=0)",
    bubble_sheet="Breakdowns",
    out_path="figures/overlapping_bubbles.png",
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    dates_df = pd.read_excel(bubble_file, sheet_name=date_sheet)
    dates_df["Date"] = pd.to_datetime(dates_df["Date"], format="%d/%m/%Y", errors="coerce")
    date_mapping = {i: date for i, date in enumerate(dates_df["Date"], start=1)}

    bubble_data = pd.read_excel(bubble_file, sheet_name=bubble_sheet)
    bubble_data = bubble_data.sort_values(by=["Firm", "Start"])

    firms = bubble_data["Firm"].unique()
    firm_positions = {firm: i for i, firm in enumerate(firms)}

    plt.figure(figsize=(12, 6))

    boom_label_added = False
    burst_label_added = False

    for _, row in bubble_data.iterrows():
        y = firm_positions[row["Firm"]]
        start_date = date_mapping.get(row["Start"])
        peak_date = date_mapping.get(row["Peak"])
        end_date = date_mapping.get(row["End"])

        if start_date is None or peak_date is None or end_date is None:
            continue

        plt.hlines(
            y,
            start_date,
            peak_date,
            colors="green",
            linewidth=2,
            label="Boom Phase" if not boom_label_added else None,
        )
        boom_label_added = True

        plt.hlines(
            y,
            peak_date,
            end_date,
            colors="red",
            linewidth=2,
            label="Burst Phase" if not burst_label_added else None,
        )
        burst_label_added = True

    plt.yticks(range(len(firms)), firms)
    plt.xticks(rotation=45)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.legend(loc="lower left")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Overlapping bubbles chart saved: {out_path}")
