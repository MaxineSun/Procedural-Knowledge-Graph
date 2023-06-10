from models import GNNModel, MLP
import torch
import pickle
import torch.nn.functional as F
import numpy as np
import argparse
import utils.parse_args as pa

def train(args):
    with open("../scratch/data/train_loader8", "rb") as fp:
        train_loader = pickle.load(fp)
    fp.close()

    with open("../scratch/data/val_loader8", "rb") as fp:
        val_loader = pickle.load(fp)
    fp.close()

    if args.train_model == "gnn":
        model = GNNModel(hidden_channels=64, data=train_loader.data)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(1, 12):
            total_loss = total_examples = 0
            for sampled_data in train_loader:
                optimizer.zero_grad()
                sampled_data.to(device)
                pred = model(sampled_data)
                ground_truth = sampled_data["mq","queries","sq"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        preds = []
        ground_truths = []
        for sampled_data in val_loader:
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(sampled_data))
                ground_truths.append(sampled_data["mq","queries","sq"].edge_label)
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        pred = np.where(pred > 0, 1, 0)
        accu = sum(ground_truth==pred)/len(pred)
        print(f"accu: {accu:.4f}")

    if args.train_model == "random guess":
        preds = np.array([])
        ground_truths = []
        for sampled_data in val_loader:
            length = len(sampled_data["mq","queries","sq"].edge_label)
            preds = np.append(preds, np.random.choice([0, 1], size=length, p=[2/3, 1/3]))
            ground_truths.append(sampled_data["mq","queries","sq"].edge_label)
        ground_truth = torch.cat(ground_truths, dim=0).numpy()
        accu = np.sum(ground_truth==preds)/len(preds)
        print(f"accu: {accu:.4f}")
    
    if args.train_model == "MLP":
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = pa.parse_args()
    train(args)