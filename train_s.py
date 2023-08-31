from models import GNNModel, MLP, GCNet, APPNPNet
import os
import glob
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import argparse
import utils.parse_args as pa
import utils.dataset_generate as DG
from torch_geometric.loader import DataLoader


def calculate_f1_score(actual_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(actual_labels == 1, predicted_labels == 1))
    false_positive = np.sum(np.logical_and(actual_labels == 0, predicted_labels == 1))
    false_negative = np.sum(np.logical_and(actual_labels == 1, predicted_labels == 0))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNet(-1, 64, 11)
    dataset = []
    # for i in range(32774):
    #     with open(f"../scratch/data/subgraphs/g{str(i).zfill(5) }", "rb") as fp:
    #         subgraph = pickle.load(fp)
    #     fp.close()
    #     dataset.append(subgraph)

    with open("../scratch/data/subgraphs/dataset_isolated", "rb") as fp:
        dataset = pickle.load(fp)
    fp.close()

    datalength = len(dataset)
    train_len = int(0.6*datalength)
    val_len = int(0.2*datalength)
    train_dataset = dataset[:train_len]
    val_dataset = dataset[train_len:train_len+val_len]
    # test_dataset = dataset[train_len+val_len:]
    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        model.train()
        for ind, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            out = torch.flatten(out)
            gt = F.one_hot(batch.y, num_classes = 11).float()
            gt = torch.flatten(gt)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(out, gt)
            loss.backward()
            optimizer.step()
            del out
        val_accu = []
        for ind, batch in enumerate(val_dataloader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            val_accu.extend(torch.max(out, dim=1)[1] == batch.y)
        accu = sum(val_accu) / len(val_accu)
        print(accu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = pa.parse_args()
    main(args)