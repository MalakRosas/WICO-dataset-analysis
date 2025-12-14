import networkx as nx
import pandas as pd
import numpy as np

def evasion_attack(G, bot_nodes, num_edges_per_bot):
    degree = dict(G.degree())
    degree_df = pd.DataFrame({"node": list(degree.keys()), "degree": list(degree.values())})

    top_influencers = (
        degree_df.sort_values(by="degree", ascending=False)
                 .head(100)["node"]   
                 .tolist()
    )
    print(f"Top influencers selected: {len(top_influencers)}")
    for bot in bot_nodes:
        selected = np.random.choice(top_influencers, num_edges_per_bot, replace=False) 
        for target in selected:
            if bot != target:
                G.add_edge(bot, target)

    print(f"Added {num_edges_per_bot} influencer edges for each bot.")
    print("Evasion completed.\n")

    return G
