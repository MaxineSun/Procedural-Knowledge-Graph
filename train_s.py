from models import GNNModel, MLP, GCNet
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import argparse
import utils.parse_args as pa
import utils.dataset_generate as DG


def calculate_f1_score(actual_labels, predicted_labels):
    true_positive = np.sum(np.logical_and(actual_labels == 1, predicted_labels == 1))
    false_positive = np.sum(np.logical_and(actual_labels == 0, predicted_labels == 1))
    false_negative = np.sum(np.logical_and(actual_labels == 1, predicted_labels == 0))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)[data.train_mask]
    out = torch.flatten(out)
    gt = F.one_hot(data.y[data.train_mask], num_classes = 11)
    gt = torch.flatten(gt)
    loss = F.nll_loss(out, gt)
    loss.backward()
    optimizer.step()
    del out

def test(model, data):
    model.eval()
    logits, accs, losses, preds = model(data.x, data.edge_index), [], [], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        loss = F.nll_loss(model(data.x, data.edge_index)[mask], data.y[mask])
        preds.append(pred.detach().cpu())
        accs.append(acc)
        losses.append(loss.detach().cpu())
    return accs, preds, losses

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNet(-1, 64, 11)
    dataset = DG.PKG_Dataset("../scratch/data/graph")
    data = dataset.get()
    
    data = data.to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = test_acc = 0
    for epoch in range(1, 2):
        train(model, optimizer, data)
        [train_acc, val_acc, tmp_test_acc], preds, [train_loss, val_loss, tmp_test_loss] = test(model, data)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = pa.parse_args()
    main(args)