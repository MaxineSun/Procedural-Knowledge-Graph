import pickle
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertForNextSentencePrediction
        

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_follow = BertForNextSentencePrediction.from_pretrained(model_name)
    model_follow = model_follow.to(device)
    with open("../scratch/data/nsg/train_dataloader_follow", "rb") as fp:
        train_dataloader_follow = pickle.load(fp)
    fp.close()
    with open("../scratch/data/nsg/val_dataloader_follow", "rb") as fp:
        val_dataloader_follow = pickle.load(fp)
    fp.close()
    # with open("../scratch/data/nsg/test_dataloader_follow", "rb") as fp:
    #     test_dataloader_follow = pickle.load(fp)
    # fp.close()

    optimizer = optim.Adam(model_follow.parameters(), lr=args.learning_rate_nsg)
    criterion = torch.nn.BCELoss()
    model_follow.train()
    for epoch in range(300):
        train_accu = []
        for item in train_dataloader_follow:
            encoded_inputs = tokenizer(list(item[0]), list(item[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model_follow(**encoded_inputs).logits)
            item[2] = item[2].to(device)
            loss = criterion(outputs.float()[:,0], item[2].float()) + criterion(outputs.float()[:,1], 1 - item[2].float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            train_accu.extend(torch.tensor(outputs).to(device)==item[2])
        train_accuracy = sum(train_accu)/len(train_accu)
        print("follow train accuracy in ", epoch, " epoch is ", train_accuracy)
        val_accu = []
        for item in val_dataloader_follow:
            encoded_inputs = tokenizer(list(item[0]), list(item[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model_follow(** encoded_inputs).logits)
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            val_accu.extend(torch.tensor(outputs)==item[2])
        val_accuracy = sum(val_accu)/len(val_accu)
        print("follow val accuracy in ", epoch, " epoch is ", val_accuracy)
    with open("../scratch/data/model/model_follow", "wb") as fp:
        pickle.dump(model_follow, fp)
    fp.close()

if __name__ == "__main__":
    args = pa.parse_args()
    train(args)
