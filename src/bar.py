import matplotlib.pyplot as plt
# Count songs per cluster
cluster_counts = df["Cluster"].value_counts().sort_index()
# Map cluster numbers → LLM-generated names
labels = [cluster_names[c] for c in cluster_counts.index]
plt.figure(figsize=(10, 5))
plt.bar(labels, cluster_counts.values)
plt.title("Song Count per Emotional Cluster")
plt.xlabel("Emotional Cluster")
plt.ylabel("Number of Songs")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
