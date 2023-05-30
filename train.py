import sys
sys.path.append("..")
from tqdm import tqdm
from models import *
import torch
import torch.nn as nn
import pickle
import numpy as np
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

    data.data.train_mask = index_to_mask(train_idx, size=data.y.shape[0])
    data.data.val_mask = index_to_mask(val_idx, size=data.y.shape[0])
    data.data.test_mask = index_to_mask(test_idx, size=data.y.shape[0])
    return data

with open("../scratch/data/processed/dataset", "rb") as fp:
    dataset = pickle.load(fp)
fp.close()
num_classes = dataset.num_classes
train_len = int(round(0.6 * len(dataset.y)))
val_len = int(round(0.2 * len(dataset.y)))
dataset = random_splits(dataset, train_len, val_len)
dataset = dataset.data.to('cuda')

model = GCNet(dataset.num_features, 32, num_classes)
model.double()
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for _ in tqdm(range(100)):
    model.train()
    optimizer.zero_grad()
    output = model(dataset.x, dataset.edge_index.T)[dataset.train_mask]
    loss = criterion(output, dataset.y[dataset.train_mask])
    loss.backward()
    optimizer.step()

model.eval()
output = model(dataset.x, dataset.edge_index.T)[dataset.train_mask]
loss = criterion(output, dataset.y[dataset.train_mask])
output_test = model(dataset.x, dataset.edge_index.T)[dataset.test_mask]
loss_test = criterion(output_test, dataset.y[dataset.test_mask])


print(loss)
print(loss_test)