import matplotlib.pyplot as plt
import numpy as np

labels = [
    "filter",
    "filter-filter",
    "filter-filter-filter",
    "filter-filter-filter-map"
]

# Updated data
lotus = [75.9, 147.5, 214.1, 317.9]
palimpzest = [75.8, 141.5, 203.6, 307.9]
ours = [77, 84.9, 92.3, 147.2]

x = np.arange(len(labels))
width = 0.15

plt.figure()

plt.bar(x - width, lotus, width, label='lotus')
plt.bar(x, palimpzest, width, label='palimpzest')
plt.bar(x + width, ours, width, label='ours')

plt.xlabel("Arxiv [Case 3]")
plt.ylabel("Seconds")
plt.title("Total Time (Seconds)")
plt.xticks(x, labels, rotation=30)
plt.legend()


plt.tight_layout()

# Save to PNG
plt.savefig("arxiv_case3.pdf", dpi=300, bbox_inches="tight")