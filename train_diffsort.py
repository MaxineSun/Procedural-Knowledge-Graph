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
    model = wikiHowNet()
    model = model.to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
    '''
    optim = torch.optim.Adam([ {'params':model.encode_model.parameters(), 'lr':args.lr_encoder}, 
                              {'params':model.fc1.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                              {'params':model.fc2.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                                # {'params':model.fc3.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}, 
                              {'params':model.act1.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6},
                              {'params':model.temperature, 'lr':args.lr_softmax, 'weight_decay':1e-6}])
                                # {'params':model.act2.parameters(), 'lr':args.lr_MLP, 'weight_decay':1e-6}])
    '''
    patience = 5
    best_val_loss = 1000.0
    current_patience = 0
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
    
        
    for epoch in range(100):
        model.train()
        loss_list = []
        loss_accu = []
        score_list = []
        for i in range(55):
            with open(f"../scratch/data/diff_sort/separate_sq_set/sq_set{str(i).zfill(5)}", "rb") as fp:
                sq_set_shuffled = pickle.load(fp)
            fp.close()
            with open(f"../scratch/data/diff_sort/separate_target_idx/target_idx{str(i).zfill(5)}", "rb") as fp:
                target_idx_set = pickle.load(fp)
            fp.close()
            
            for b_ind, (shuffle_sq, target_idx) in enumerate(zip(sq_set_shuffled, target_idx_set)):
                if target_idx_set[b_ind].size()[0] < 33:
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

                    
                    if ((b_ind + 1) % args.NUM_ACCUMULATION_STEPS == 0) or (b_ind + 1 == len(sq_set_shuffled)):
                        optim.step()
                        optim.zero_grad()
                        loss_accu.append(sum(loss_list)/len(loss_list))
                        loss_list = []
            if i % 10 == 0:
                print("shuffle_scalars: ", torch.argsort(shuffle_scalars))
                print("perm_squence: ", torch.argsort(perm_squence))
                print(score)
        print("Loss in epoch", epoch, " is ", sum(loss_accu)/len(loss_accu))
        print("Score in epoch", epoch, " is ", sum(score_list)/len(score_list))
        
            
        model.eval()
        with torch.no_grad():
            loss_list = []
            score_list = []
            for i in range(55, 73):
                with open(f"../scratch/data/diff_sort/separate_sq_set/sq_set{str(i).zfill(5)}", "rb") as fp:
                    sq_set_shuffled = pickle.load(fp)
                fp.close()
                with open(f"../scratch/data/diff_sort/separate_target_idx/target_idx{str(i).zfill(5)}", "rb") as fp:
                    target_idx_set = pickle.load(fp)
                fp.close()
                for b_ind, (shuffle_sq, target_idx) in enumerate(zip(sq_set_shuffled, target_idx_set)):
                    if target_idx_set[b_ind].size()[0] < 33:
                        shuffle_sq = shuffle_sq.to(args.device)
                        target_idx = target_idx.to(args.device)
                        shuffle_scalars, perm_prediction = model(shuffle_sq)
                        loss = torch.nn.BCELoss()(perm_prediction, target_idx)
                        loss_list.append(loss)
                        perm_squence = torch.matmul(shuffle_scalars.unsqueeze(1), target_idx)
                        score = score_function(perm_squence)
                        score_list.append(score)
                val_loss_mean = sum(loss_list)/len(loss_list)
                print("shuffle_scalars: ", torch.argsort(shuffle_scalars))
                print("perm_squence: ", torch.argsort(perm_squence))
                print(score)
            print("Val loss in epoch", epoch, " is ", val_loss_mean)
            print("Val score in epoch", epoch, " is ", sum(score_list)/len(score_list))

        if val_loss_mean <= best_val_loss:
            best_val_loss = val_loss_mean
            current_patience = 0
        else:
            current_patience += 1
            if current_patience >= patience:
                print("Early stopping triggered!")
                with open(f"../scratch/data/diff_sort/model32", "wb") as fp:
                    pickle.dump(model, fp)
                fp.close()
                break
        with open(f"../scratch/data/diff_sort/model32", "wb") as fp:
            pickle.dump(model, fp)
        fp.close()

if __name__ == "__main__":
    args = pa.parse_args()
    train(args)