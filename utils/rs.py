import torch
import numpy as np

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def random_splits(data, train_len, val_len, seed=42):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    train_len = int(round(0.6 * len(data.y)))
    train_idx = rnd_state.choice(index, train_len, replace=False)
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_len, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.edge_index = data.edge_index.T
    data.train_mask = index_to_mask(train_idx, size=data.y.shape[0])
    data.val_mask = index_to_mask(val_idx, size=data.y.shape[0])
    data.test_mask = index_to_mask(test_idx, size=data.y.shape[0])
    return data