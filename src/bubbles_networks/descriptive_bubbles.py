import os

import matplotlib.pyplot as plt
import pandas as pd


def run_descriptive_bubble_analysis(bubble_file="data/ro/ResultResults_ro_bet_bubbles.xlsx", bubble_sheet="Breakdowns"):
    os.makedirs("figures", exist_ok=True)

    bubble_res = pd.read_excel(bubble_file, sheet_name=bubble_sheet)

    vcounts = bubble_res["Firm"].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(vcounts.index, vcounts.values, color="black")
    plt.xticks(rotation=45, fontname="Comic Sans MS", fontsize=7)
    plt.xlabel("Companies")
    plt.ylabel("No. of bubbles")
    plt.savefig("figures/NoOfBubbles.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.hist(bubble_res["Duration"], bins=10)
    plt.xticks(rotation=45, fontname="Comic Sans MS", fontsize=7)
    plt.xlabel("Bubble episodes")
    plt.xticks([])
    plt.ylabel("Duration of Bubbles")
    plt.savefig("figures/histDuration.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Descriptive bubble plots saved.")
