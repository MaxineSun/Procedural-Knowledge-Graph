import sys
sys.path.append("..")
from tqdm import tqdm
from models import GCNModel
import torch
import torch.nn as nn
import pickle
import numpy as np
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask


def random_splits(data, num_classes, percls_trn, val_lb, seed=42):
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.y.shape[0])
    data.val_mask = index_to_mask(val_idx, size=data.y.shape[0])
    data.test_mask = index_to_mask(test_idx, size=data.y.shape[0])
    return data

with open("../../scratch/data/dataset_light", "rb") as fp:
    dataset = pickle.load(fp)
fp.close()


percls_trn = int(round(0.8 * len(dataset.y)))
val_lb = int(round(0.2 * len(dataset.y)))
dataset = random_splits(dataset, dataset.num_classes, percls_trn, val_lb)
labels = torch.zeros(dataset.y.shape[0], dataset.y.shape[0])
for i in range(dataset.y.shape[0]):
    labels[i][dataset.y[i]]=1
dataset.data.to('cuda')
labels.to('cuda')

model = GCNModel(-1, 64, dataset.y.shape[0])
model.double()
model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


for _ in tqdm(range(200)):
    model.train()
    optimizer.zero_grad()
    output = model(dataset.x, dataset.edge_index)[:,dataset.train_mask]
    loss = torch.nn.functional.nll_loss(output, labels[:, dataset.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
output = model(dataset.x, dataset.edge_index)[dataset.train_mask]
# predicted_labels = output.argmax(dim=1)
print(output)
print(dataset.y[dataset.train_mask])