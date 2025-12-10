from sklearn.manifold import TSNE
import plotly.express as px

embedding_matrix = np.vstack(df["Embedding"].to_numpy())
tsne = TSNE(n_components = 2, random_state = 42, perplexity = 5)
tsne_results = tsne.fit_transform(embedding_matrix)
df["TSNE_1"] = tsne_results[:, 0]
df["TSNE_2"] = tsne_results[:, 1]

#plot TF-IDF vector space with color-coded clusters
fig = px.scatter(
    df,
    x="TSNE_1",
    y="TSNE_2",
    color="Cluster Names",
    hover_data=["Title", "Cluster Names"],
    title="t-SNE Visualization of Song Emotion Clusters",
    height=780,
    width=920
)

#fix title position
fig.update_layout(title_text="Stray Kids Discography Cluster Map (TF-IDF vector projection, color-coded by cluster)", title_x=0.5)
fig.show()
