import pickle
import utils.parse_args as pa
import torch
import torch.nn as nn
from tqdm import tqdm
from models import wikiHowNet
from utils import functional
import matplotlib.pyplot as plt


def avg(list_result):
    if len(list_result) == 0:
        return 0
    return sum(list_result)/len(list_result)

def main(args):
    with open("../scratch/data/diff_sort/model", "rb") as fp:
        model = pickle.load(fp)
    fp.close()
    model = model.to(args.device)
    model_ori = wikiHowNet()
    model_ori = model_ori.to(args.device)

    if args.score_type == "inversions":
        score_function = functional.score_inversions
    if args.score_type == "emd":
        score_function = functional.score_emd
    
    model.eval()
    result_loss_list = [[] for _ in range(100)]
    result_score_list = [[] for _ in range(100)]
    random_result_loss_list = [[] for _ in range(100)]
    random_result_score_list = [[] for _ in range(100)]
    with torch.no_grad():
        for i in tqdm(range(92)):
            with open(f"../scratch/data/diff_sort/separate_sq_set/sq_set{str(i).zfill(5)}", "rb") as fp:
                sq_set_shuffled = pickle.load(fp)
            fp.close()
            with open(f"../scratch/data/diff_sort/separate_target_idx/target_idx{str(i).zfill(5)}", "rb") as fp:
                target_idx_set = pickle.load(fp)
            fp.close()
            for b_ind, (shuffle_sq, target_idx) in enumerate(zip(sq_set_shuffled, target_idx_set)):
                shuffle_sq = shuffle_sq.to(args.device)
                target_idx = target_idx.to(args.device)
                length = target_idx_set[b_ind].size()[0]
                shuffle_scalars, perm_prediction = model(shuffle_sq)
                loss = torch.nn.BCELoss()(perm_prediction, target_idx)
                result_loss_list[length].append(loss)
                perm_squence = torch.matmul(shuffle_scalars.unsqueeze(1), target_idx)
                # print(perm_squence)
                score = score_function(perm_squence)
                result_score_list[length].append(score)
                
                shuffle_scalars_random, perm_prediction_random = model_ori(shuffle_sq)
                loss_random = torch.nn.BCELoss()(perm_prediction_random, target_idx)
                random_result_loss_list[length].append(loss_random)
                perm_squence_random = torch.matmul(shuffle_scalars_random.unsqueeze(1), target_idx)
                score_random = score_function(perm_squence_random)
                random_result_score_list[length].append(score_random)
                
    with open(f"../scratch/data/diff_sort/dist_stat/result_loss_list", "wb") as fp:
        pickle.dump(result_loss_list, fp)
    fp.close()
    with open(f"../scratch/data/diff_sort/dist_stat/result_score_list", "wb") as fp:
        pickle.dump(result_score_list, fp)
    fp.close()
    with open(f"../scratch/data/diff_sort/dist_stat/random_result_loss_list", "wb") as fp:
        pickle.dump(random_result_loss_list, fp)
    fp.close()
    with open(f"../scratch/data/diff_sort/dist_stat/random_result_score_list", "wb") as fp:
        pickle.dump(random_result_score_list, fp)
    fp.close()


if __name__ == "__main__":
    # args = pa.parse_args()
    # main(args)
    with open("../scratch/data/diff_sort/dist_stat/result_loss_list", "rb") as fp:
        result_loss_list = pickle.load(fp)
    with open("../scratch/data/diff_sort/dist_stat/result_score_list", "rb") as fp:
        result_score_list = pickle.load(fp)
    with open("../scratch/data/diff_sort/dist_stat/random_result_loss_list", "rb") as fp:
        random_result_loss_list = pickle.load(fp)
    with open("../scratch/data/diff_sort/dist_stat/random_result_score_list", "rb") as fp:
        random_result_score_list = pickle.load(fp)    
        
    avg_result_loss_list = [float(avg(l)) for l in result_loss_list]
    avg_result_score_list = [float(avg(l)) for l in result_score_list]
    avg_random_result_loss_list = [float(avg(l)) for l in random_result_loss_list]
    avg_random_result_score_list = [float(avg(l)) for l in random_result_score_list]

    x_list = list(range(20))
    plt.plot(x_list, avg_result_loss_list[:20], label='loss')
    plt.plot(x_list, avg_result_score_list[:20], label='score')
    plt.plot(x_list, avg_random_result_loss_list[:20], label='random loss')
    plt.plot(x_list, avg_random_result_score_list[:20], label='random score')

    plt.title('result distribution')
    plt.xlabel('sentence length')
    plt.ylabel('loss or score')

    plt.legend()
    plt.savefig('h20.png')
    
    
        
    
    