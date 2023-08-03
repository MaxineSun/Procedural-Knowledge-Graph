import pickle
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.utils.data import DataLoader


def train(args):
    with open(f"../scratch/data/rw/random_walk_data1k", "rb") as fp:
        data = pickle.load(fp)
    fp.close()
    dataset_size = len(data.score)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    with open("../scratch/data/nsg/val_dataloader", "rb") as fp:
        val_dataloader = pickle.load(fp)
    fp.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    with open("../scratch/data/model/model15", "rb") as fp:
        model = pickle.load(fp)
    fp.close()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_nsg, weight_decay=1e-5)

    
    for epoch in range(20):
        for ind, pair in enumerate(train_dataset):
            model.train()
            encoded_inputs = tokenizer(pair[0][0], pair[0][1], return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model(**encoded_inputs).logits)
            outputs = torch.nn.functional.normalize(outputs, p=1.0, dim=1, eps=1e-12, out=None)
            target = torch.tensor(float(pair[1])).to(device)
            loss = F.mse_loss(outputs[:,0].prod(), target) + F.mse_loss(outputs[:,1].prod(), 1 - target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ind % 64 == 0:
                model.eval()
                val_accu = []
                for batch in val_dataloader:
                    encoded_inputs = tokenizer(list(batch[0]), list(batch[1]), return_tensors='pt', padding=True, truncation=True).to(device)
                    outputs = torch.sigmoid(model(** encoded_inputs).logits)
                    outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
                    val_accu.extend(torch.tensor(outputs)==batch[2])
                val_accuracy = sum(val_accu)/len(val_accu)
                print("val accuracy in ", epoch, " epoch ", ind//64," batch is ", val_accuracy)
    with open("../scratch/data/model/model15", "wb") as fp:
        pickle.dump(model, fp)
    fp.close()

if __name__ == "__main__":
    args = pa.parse_args()
    train(args)
