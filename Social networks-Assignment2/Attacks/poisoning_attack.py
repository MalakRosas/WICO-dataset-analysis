import numpy as np
import pandas as pd
import torch

def poisoning_attack(df, train_mask, flip_ratio=0.3):
    train_indices = np.where(train_mask.numpy())[0]
    
    bots_in_train = df.iloc[train_indices]
    bots_in_train = bots_in_train[bots_in_train["is_bot"] == 1]

    num_bots = len(bots_in_train)
    num_to_flip = int(num_bots * flip_ratio)

    print(f"Total bots in train set: {num_bots}")
    print(f"Flipping labels for {num_to_flip} bots")

    flipped_indices = np.random.choice(bots_in_train.index, num_to_flip, replace=False)

    df.loc[flipped_indices, "is_bot"] = 0
    return df
