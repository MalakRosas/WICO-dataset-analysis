import sys, os
sys.path.append(os.path.abspath(".."))

import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from node2vec import Node2Vec
from networkx.algorithms.community import louvain_communities
import matplotlib.pyplot as plt
from Attacks.poisoning_attack import poisoning_attack

file_path = "../facebook_combined.txt/facebook_combined.txt"
G = nx.read_edgelist(file_path, nodetype=int)

df = pd.read_csv("bots_detected.csv")
bot_nodes = df[df["is_bot"] == 1]["node"].tolist()
print(f"\nLoaded {len(bot_nodes)} bots from detect_bots.py")


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


node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=200, workers=4)
model_n2v = node2vec.fit(window=10, min_count=1)
embeddings = [model_n2v.wv[str(n)] for n in df["node"]]
emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(64)])
df = pd.concat([df, emb_df], axis=1)

feature_cols = [
    "degree", "clustering", "eigenvector",
    "betweenness", "closeness", "community"
] + [f"emb_{i}" for i in range(64)]

X = torch.tensor(df[feature_cols].values, dtype=torch.float)
df["is_bot_original"] = df["is_bot"]  
y = torch.tensor(df["is_bot"].values, dtype=torch.long)
data = from_networkx(G)
data.x = X
data.y = y
data.num_features = X.shape[1]

num_nodes = data.num_nodes
perm = torch.randperm(num_nodes)
train_size = int(0.7 * num_nodes)

train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_mask[perm[:train_size]] = True
test_mask[perm[train_size:]] = True

data.train_mask = train_mask
data.test_mask = test_mask

# (label flipping)
df_poisoned = df.copy()
df_poisoned = poisoning_attack(df_poisoned, train_mask, flip_ratio=0.3)

# Update labels in graph data
data.y = torch.tensor(df_poisoned["is_bot"].values, dtype=torch.long)

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden)
        self.conv2 = SAGEConv(hidden, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


model = GraphSAGE(data.num_features, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


print("\nTraining GraphSAGE on poisoned data")
for epoch in range(40):
    model.train()
    optimizer.zero_grad()

    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

model.eval()
logits = model(data)
pred = logits.argmax(dim=1)
y_true = data.y[data.test_mask]
y_pred = pred[data.test_mask]

print("\nPOISONING ATTACK RESULTS")
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))

os.makedirs("plots", exist_ok=True)
pos = nx.spring_layout(G, seed=42)
node_colors = []
for idx, row in df_poisoned.iterrows():
    if row["is_bot_original"] == 1 and row["is_bot"] == 0:
        node_colors.append("yellow")   # flipped bots
    elif row["is_bot"] == 1:
        node_colors.append("red")      # bots
    else:
        node_colors.append("blue")     # humans

plt.figure(figsize=(12, 12))
nx.draw(
    G,
    pos,
    node_color=node_colors,
    node_size=10,
    edge_color="gray",
    alpha=0.6
)
plt.title("Poisoning Attack Graph (Yellow = Label-Flipped Bots)")
plt.savefig("plots/poisoning_graph.png", dpi=300)
plt.close()