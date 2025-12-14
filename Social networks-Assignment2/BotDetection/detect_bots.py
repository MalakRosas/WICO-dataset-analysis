import networkx as nx
import pandas as pd

file_path = "../facebook_combined.txt/facebook_combined.txt"
G = nx.read_edgelist(file_path, nodetype=int)

degree = dict(G.degree())
clustering = nx.clustering(G)
eigen = nx.eigenvector_centrality(G, max_iter=500)
between = nx.betweenness_centrality(G, k=500)
close = nx.closeness_centrality(G)

df = pd.DataFrame({
    "node": list(G.nodes()),
    "degree": [degree[n] for n in G.nodes()],
    "clustering": [clustering[n] for n in G.nodes()],
    "eigenvector": [eigen[n] for n in G.nodes()],
    "betweenness": [between[n] for n in G.nodes()],
    "closeness": [close[n] for n in G.nodes()],
})

avg_degree = df["degree"].mean()
low_clust = df["clustering"].quantile(0.20)      
low_eigen = df["eigenvector"].quantile(0.20)   
low_betw = df["betweenness"].quantile(0.20)     
low_close = df["closeness"].quantile(0.20)    

df["is_bot"] = (
    ((df["degree"] > avg_degree) & (df["clustering"] < low_clust)) | 
    ((df["eigenvector"] < low_eigen) & (df["betweenness"] < low_betw))
).astype(int)

bot_nodes = df[df["is_bot"] == 1]["node"].tolist() 

print("\nBots detected:", len(bot_nodes))
print(bot_nodes[:30])

df.to_csv("bots_detected.csv", index=False)
print("\nSaved detected bots to bots_detected.csv")
