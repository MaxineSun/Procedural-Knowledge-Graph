import pickle
import copy
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from models import wikiHowNet
from utils import diffsort


def sorter(size):
    sorter = diffsort.DiffSortNet(
        sorting_network_type=args.method,
        size=size,
        device=args.device,
        steepness=args.steepness,
        art_lambda=args.art_lambda,
    )
    return sorter

def train(args):
    model = wikiHowNet()
    model = model.to(args.device)
    # model_t = wikiHowNet()
    # model_t = model.to(args.device)

    optim = torch.optim.Adam(model.parameters(), lr=10**(-4))
    criterion = torch.nn.MSELoss()
    model.train()

    with open("../scratch/data/diff_sort/train_dataloader", "rb") as fp:
        train_dataloader = pickle.load(fp)
    fp.close()

    # with open("../scratch/data/diff_sort/val_dataloader", "rb") as fp:
    #     val_dataloader = pickle.load(fp)
    # fp.close()

    # with open("../scratch/data/diff_sort/test_dataloader", "rb") as fp:
    #     test_dataloader = pickle.load(fp)
    # fp.close()

    sorter = diffsort.DiffSortNet(
        sorting_network_type='bitonic',
        size=args.num_compare,
        device=args.device,
        steepness=args.steepness,
        art_lambda=args.art_lambda,
    )
    num_batches = len(train_dataloader)
    for epoch in range(3):
        model.train()
        for b_ind, batch in tqdm(enumerate(train_dataloader)):
            if b_ind < num_batches - 1:
                batch[0] = batch[0].to(args.device)
                batch[1] = batch[1].to(args.device)
                shuffle_scalars = model(batch[0]).squeeze(2)
                shuffle_scalars, _ = sorter(shuffle_scalars)
                target_scalars = model(batch[1]).squeeze(2)
                # target_scalars = torch.tensor([[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0]]*32).to(args.device)
                loss = criterion(shuffle_scalars, target_scalars)
                print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()


if __name__ == "__main__":
    args = pa.parse_args()
    train(args)