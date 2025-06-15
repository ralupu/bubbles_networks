import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
bubble_res = pd.read_excel('data_ro/ResultResults_ro_bet_bubbles.xlsx', sheet_name='Breakdowns')
#%%
# bubble_res.describe()
#%%
vcounts = bubble_res['Firm'].value_counts()
#%%
plt.figure(figsize=(12, 6))
plt.bar(vcounts.index, vcounts.values, color='black')

# Set the font and rotation for x-axis labels
plt.xticks(rotation=45, fontname='Comic Sans MS', fontsize=7)  # Rotate labels to 45 degrees, change font and size

plt.xlabel('Companies')  # X-axis label
plt.ylabel('No. of bubbles')  # Y-axis label
# plt.title('Sample Bar Plot')  # Plot title
# plt.show()
plt.savefig('figures/NoOfBubbles.png')

# bubble_res['Firm'].value_counts().to_excel('results/interim/bubbles_all_value_counts.xlsx')
# desc_stats = bubble_res.describe()
# desc_stats.to_excel('results/interim/bubbles_desc_stats.xlsx')
#
# bubbles_per_comp = bubble_res['Firm'].value_counts()

plt.figure(figsize=(12, 6))
plt.hist(bubble_res['Duration'], bins=10)

# Set the font and rotation for x-axis labels
plt.xticks(rotation=45, fontname='Comic Sans MS', fontsize=7)  # Rotate labels to 45 degrees, change font and size

plt.xlabel('Bubble episodes')  # X-axis label
plt.xticks([])
plt.ylabel('Duration of Bubbles')  # Y-axis label
# plt.title('Sample Bar Plot')  # Plot title
# plt.show()
plt.savefig('figures/histDuration.png')

