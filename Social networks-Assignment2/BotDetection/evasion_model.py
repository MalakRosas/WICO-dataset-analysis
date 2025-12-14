import sys, os
sys.path.append(os.path.abspath(".."))

import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
from sklearn.metrics import accuracy_score, classification_report
from node2vec import Node2Vec
from networkx.algorithms.community import louvain_communities
from Attacks.evasion_attack import evasion_attack
import matplotlib.pyplot as plt

file_path = "../facebook_combined.txt/facebook_combined.txt"
G = nx.read_edgelist(file_path, nodetype=int)

df = pd.read_csv("bots_detected.csv")
bot_nodes = df[df["is_bot"] == 1]["node"].tolist()
print(f"\nLoaded {len(bot_nodes)} bots from detect_bots.py")

G = evasion_attack(G, bot_nodes, num_edges_per_bot=7)

degree = dict(G.degree())
clustering = nx.clustering(G)
eigen = nx.eigenvector_centrality(G, max_iter=500)
between = nx.betweenness_centrality(G, k=500)
close = nx.closeness_centrality(G)

communities = louvain_communities(G)
community_map = {}
for i, comm in enumerate(communities):
    for node in comm:
        community_map[node] = i

df["community"] = df["node"].apply(lambda x: community_map[x])


node_order = pd.read_csv("models/node_order.csv")["node"].tolist()
df = df.set_index("node").loc[node_order].reset_index()

# (graph changed)
node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, workers=4)
n2v_model = node2vec.fit(window=10, min_count=1)

embeddings = [n2v_model.wv[str(n)] for n in df["node"]]
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(64)])
df = pd.concat([df, emb_df], axis=1)

with open("models/feature_columns.txt") as f:
    feature_cols = [line.strip() for line in f.readlines()]

X = torch.tensor(df[feature_cols].values, dtype=torch.float)
y = torch.tensor(df["is_bot"].values, dtype=torch.long)

data = from_networkx(G)
data.x = X
data.y = y

data.train_mask = torch.load("models/train_mask.pt")
data.test_mask = torch.load("models/test_mask.pt")

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


model = GraphSAGE(len(feature_cols), 64, 2)
model.load_state_dict(torch.load("models/baseline_model.pth"))
model.eval()


logits = model(data)
pred = logits.argmax(dim=1)

y_true = data.y[data.test_mask]
y_pred = pred[data.test_mask]

print("\n EVASION ATTACK RESULTS ")
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))


os.makedirs("plots", exist_ok=True)
pos = nx.spring_layout(G, seed=42)
node_colors = [
    "red" if df.loc[df["node"] == n, "is_bot"].values[0] == 1 else "blue"
    for n in G.nodes()
]

plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos,
    node_color=node_colors,
    node_size=10,
    edge_color="gray",
    alpha=0.6
)

plt.title("Graph After Evasion Attack (Bots in Red, Humans in Blue)")
plt.savefig("plots/evasion_graph.png", dpi=300)
plt.close()