import pickle
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertForNextSentencePrediction
        

def train(args):
    with open("../scratch/data/nsg/train_dataloader_head", "rb") as fp:
        train_dataloader_head = pickle.load(fp)
    fp.close()
    with open("../scratch/data/nsg/val_dataloader_head", "rb") as fp:
        val_dataloader_head = pickle.load(fp)
    fp.close()
    # with open("../scratch/data/nsg/test_dataloader_head", "rb") as fp:
    #     test_dataloader_head = pickle.load(fp)
    # fp.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_head = BertForNextSentencePrediction.from_pretrained(model_name)
    model_head = model_head.to(device)
    optimizer = optim.Adam(model_head.parameters(), lr=args.learning_rate_nsg, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    model_head.train()
    for epoch in range(300):
        train_accu = []
        for item in train_dataloader_head:
            encoded_inputs = tokenizer(list(item[0]), list(item[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model_head(**encoded_inputs).logits)
            item[2] = item[2].float().to(device)
            loss = criterion(outputs[:,0], item[2]) + criterion(outputs[:,1], 1 - item[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            train_accu.extend(torch.tensor(outputs).to(device)==item[2])
        train_accuracy = sum(train_accu)/len(train_accu)
        print("head train accuracy in ", epoch, " epoch is ", train_accuracy)
        val_accu = []
        for item in val_dataloader_head:
            encoded_inputs = tokenizer(list(item[0]), list(item[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model_head(** encoded_inputs).logits)
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            val_accu.extend(torch.tensor(outputs)==item[2])
        val_accuracy = sum(val_accu)/len(val_accu)
        print("head val accuracy in ", epoch, " epoch is ", val_accuracy)
    with open("../scratch/data/model/model_head", "wb") as fp:
        pickle.dump(model_head, fp)
    fp.close()


if __name__ == "__main__":
    args = pa.parse_args()
    train(args)
