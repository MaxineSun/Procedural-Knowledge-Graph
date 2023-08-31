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
    model = BertForNextSentencePrediction.from_pretrained(model_name)
    model = model.to(device)
    with open("../scratch/data/nsg/train_dataloader", "rb") as fp:
        train_dataloader = pickle.load(fp)
    fp.close()
    with open("../scratch/data/nsg/val_dataloader", "rb") as fp:
        val_dataloader = pickle.load(fp)
    fp.close()
    # with open("../scratch/data/nsg/test_dataloader", "rb") as fp:
    #     test_dataloader_follow = pickle.load(fp)
    # fp.close()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_nsg, weight_decay=1e-5)
    criterion = torch.nn.BCELoss()
    model.train()
    for epoch in range(15):
        train_accu = []
        for batch in train_dataloader:
            encoded_inputs = tokenizer(list(batch[0]), list(batch[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model(**encoded_inputs).logits)
            batch[2] = batch[2].float().to(device)
            loss = criterion(outputs[:,0], batch[2]) + criterion(outputs[:,1], 1 - batch[2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            train_accu.extend(torch.tensor(outputs).to(device)==batch[2])
        train_accuracy = sum(train_accu)/len(train_accu)
        print("mixed train accuracy in ", epoch, " epoch is ", train_accuracy)
        val_accu = []
        for batch in val_dataloader:
            encoded_inputs = tokenizer(list(batch[0]), list(batch[1]), return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model(** encoded_inputs).logits)
            outputs = [0 if item < 0.5 else 1 for item in outputs[:,0]]
            val_accu.extend(torch.tensor(outputs)==batch[2])
        val_accuracy = sum(val_accu)/len(val_accu)
        print("mixed val accuracy in ", epoch, " epoch is ", val_accuracy)
    with open("../scratch/data/model/model15", "wb") as fp:
        pickle.dump(model, fp)
    fp.close()
    # with open("../scratch/data/nsg/test_shuffled", "rb") as fp:
    #     test_shuffled = pickle.load(fp)
    # fp.close()

if __name__ == "__main__":
    args = pa.parse_args()
    train(args)
