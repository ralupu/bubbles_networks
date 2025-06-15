import pandas as pd
import matplotlib.pyplot as plt


def run_descriptive_bubble_analysis():
    # Load bubble results
    bubble_res = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')

    # Plot: Number of bubbles per firm
    vcounts = bubble_res['Firm'].value_counts()
    plt.figure(figsize=(12, 6))
    plt.bar(vcounts.index, vcounts.values, color='black')
    plt.xticks(rotation=45, fontname='Comic Sans MS', fontsize=7)
    plt.xlabel('Companies')
    plt.ylabel('No. of bubbles')
    plt.savefig('figures/NoOfBubbles.png')
    plt.close()

    # Plot: Histogram of bubble durations
    plt.figure(figsize=(12, 6))
    plt.hist(bubble_res['Duration'], bins=10)
    plt.xticks(rotation=45, fontname='Comic Sans MS', fontsize=7)
    plt.xlabel('Bubble episodes')
    plt.xticks([])
    plt.ylabel('Duration of Bubbles')
    plt.savefig('figures/histDuration.png')
    plt.close()

    print("✅ Descriptive bubble plots saved.")
