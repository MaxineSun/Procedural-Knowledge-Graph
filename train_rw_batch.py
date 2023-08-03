import pickle
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertTokenizer, BertForNextSentencePrediction
from torch.utils.data import DataLoader


def mse(output, target):
    output = output.tolist()
    target = target.tolist()
    squared_diff = [(x - y) ** 2 for x, y in zip(output, target)]
    mse = sum(squared_diff) / len(target)
    return mse


def create_lists(batch, batch_size, walk_len):
    input_sentence = []
    target_sentence = []
    for batch_ind in range(batch_size):
        for walk_ind in range(walk_len):
            input_sentence.append(batch[0][0][walk_ind][batch_ind])
            target_sentence.append(batch[0][1][walk_ind][batch_ind])
    return input_sentence, target_sentence, batch[1]

def output_cal(output, batch_size, walk_len, device):
    agg_output_p = torch.zeros(batch_size, requires_grad=True).to(device)
    agg_output_n = torch.zeros(batch_size, requires_grad=True).to(device)
    for walk_ind in range(batch_size):
        agg_output_p[walk_ind] = output[6*walk_ind:6*(1+walk_ind),0].prod()
        agg_output_n[walk_ind] = output[6*walk_ind:6*(1+walk_ind),1].prod()
    return agg_output_p, agg_output_n

def train(args):
    torch.cuda.empty_cache()
    with open(f"../scratch/data/rw/random_walk_data", "rb") as fp:
        data = pickle.load(fp)
    fp.close()
    dataset_size = len(data.score)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.nsg_batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.nsg_batch_size, shuffle=True, drop_last=True)
    with open("../scratch/data/nsg/val_dataloader", "rb") as fp:
        val_dataloader_p = pickle.load(fp)
    fp.close()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    with open("../scratch/data/model/model15", "rb") as fp:
        model = pickle.load(fp)
    fp.close()
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate_nsg, weight_decay=1e-5)

    for epoch in range(100):
        train_loss = []
        model.train()
        model = model.to(device)
        for ind, batch in enumerate(train_dataloader):
            input_sentence, target_sentence, scores = create_lists(batch, len(batch[1]), 6)
            encoded_inputs = tokenizer(input_sentence, target_sentence, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model(**encoded_inputs).logits)
            outputs = torch.nn.functional.normalize(outputs, p=1.0, dim=1, eps=1e-12, out=None)
            agg_output_p, _ = output_cal(outputs, args.nsg_batch_size, 6, device)
            scores = scores.float().to(device)
            loss = criterion(agg_output_p, scores) # + criterion(agg_output_n, 1 - scores)
            train_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del encoded_inputs, outputs, agg_output_p, scores, loss
        train_avg_loss = torch.mean(torch.tensor(train_loss))
        print("train loss in ", epoch, " epoch is ", train_avg_loss)
        torch.cuda.empty_cache()

        model.eval()
        val_loss = []
        for batch in val_dataloader:
            input_sentence, target_sentence, scores = create_lists(batch, len(batch[1]), 6)
            encoded_inputs = tokenizer(input_sentence, target_sentence, return_tensors='pt', padding=True, truncation=True).to(device)
            outputs = torch.sigmoid(model(**encoded_inputs).logits)
            outputs = torch.nn.functional.normalize(outputs, p=1.0, dim=1, eps=1e-12, out=None)
            agg_output_p, _ = output_cal(outputs, args.nsg_batch_size, 6, device)
            scores = scores.float().to(device)
            loss = mse(agg_output_p, scores)
            val_loss.append(loss)
            del encoded_inputs, outputs, agg_output_p, scores, loss
            torch.cuda.empty_cache()
        val_avg_loss = torch.mean(torch.tensor(val_loss))
        print("val loss in ", epoch, " epoch is ", val_avg_loss)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = pa.parse_args()
    train(args)
