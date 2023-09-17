import pickle
import utils.parse_args as pa
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from models import wikiHowNet
from utils import functional
import numpy as np
from scipy.stats import wasserstein_distance


def train(args):
    with open("../scratch/data/diff_sort/sq_set_shuffled", "rb") as fp:
        sq_set_shuffled = pickle.load(fp)
    fp.close()
    with open("../scratch/data/diff_sort/target_idx", "rb") as fp:
        target_idx_set = pickle.load(fp)
    fp.close()
    sq_set_shuffled_val = sq_set_shuffled[30_720:40_960]
    target_idx_set_val = target_idx_set[30_720:40_960]
    sq_set_shuffled = sq_set_shuffled[:30_720]
    target_idx_set = target_idx_set[:30_720]
    
    model = wikiHowNet()
    model = model.to(args.device)
    optim = torch.optim.Adam([{'params':model.encode_model.parameters(), 'lr':args.lr_encoder}, 
                              {'params':model.fc1.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                              {'params':model.fc2.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                                # {'params':model.fc3.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                              {'params':model.act1.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6},])
                            #    {'params':model.act2.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}])
    
    patience = 5
    best_val_loss = 1000.0
    current_patience = 0
    count = 0
    if args.score_type == "inversions":
        score_function = functional.score_inversions
    if args.score_type == "emd":
        score_function = functional.score_emd
    if args.score_type == "si-snr":
        score_function = functional.score_si_snr
    if args.score_type == "pesq":
        score_function = functional.score_pesq
    if args.score_type == "pit":
        score_function = functional.score_pit
    if args.score_type == "eed":
        score_function = functional.score_eed
    
        
    for epoch in range(100):
        model.train()
        loss_list = []
        loss_accu = []
        score_list = []
        for b_ind, (shuffle_sq, target_idx) in enumerate(zip(sq_set_shuffled, target_idx_set)):
            shuffle_sq = shuffle_sq.to(args.device)
            target_idx = target_idx.to(args.device)
            shuffle_scalars, perm_prediction = model(shuffle_sq)
            loss = torch.nn.BCELoss()(perm_prediction, target_idx)
            loss_list.append(loss)
            # print("loss: ", loss)
            loss = loss / args.NUM_ACCUMULATION_STEPS
            loss.backward()
            perm_squence = torch.matmul(shuffle_scalars.unsqueeze(1), target_idx)
            score = score_function(perm_squence)
            score_list.append(score)
            if loss <0.5 and count <4 and len(perm_squence) >7:
                count +=1
                print("shuffle_scalars: ", shuffle_scalars)
                print("perm_squence: ", perm_squence)
            # print("score: ", score)
            if ((b_ind + 1) % args.NUM_ACCUMULATION_STEPS == 0) or (b_ind + 1 == len(sq_set_shuffled)):
                optim.step()
                optim.zero_grad()
                # print("Loss in epoch", epoch, "batch", (b_ind + 1) // args.NUM_ACCUMULATION_STEPS, "is ", sum(loss_list)/len(loss_list))
                loss_accu.append(sum(loss_list)/len(loss_list))
                loss_list = []
        print("Loss in epoch", epoch, " is ", sum(loss_accu)/len(loss_accu))
        print("Score in epoch", epoch, " is ", sum(score_list)/len(score_list))
        
        
        model.eval()
        with torch.no_grad():
            loss_list = []
            score_list = []
            for b_ind, (shuffle_sq, target_idx) in enumerate(zip(sq_set_shuffled_val, target_idx_set_val)):
                shuffle_sq = shuffle_sq.to(args.device)
                target_idx = target_idx.to(args.device)
                perm_squence, perm_prediction = model(shuffle_sq)
                loss = torch.nn.BCELoss()(perm_prediction, target_idx)
                loss_list.append(loss)
                score = score_function(perm_squence)
                score_list.append(score)
            val_loss_mean = sum(loss_list)/len(loss_list)
            print("Val loss in epoch", epoch, " is ", val_loss_mean)
            print("Val score in epoch", epoch, " is ", sum(score_list)/len(score_list))

        if val_loss_mean <= best_val_loss:
            best_val_loss = val_loss_mean
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                print("Early stopping triggered!")
                break

if __name__ == "__main__":
    args = pa.parse_args()
    train(args)