import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import functional


class SentenceDataset(Dataset):
    def __init__(self,shuffled_embs, target_idx_set):
        self.shuffled_embs = shuffled_embs
        self.target_idx_set = target_idx_set

    def __len__(self):
        return len(self.target_idx_set)

    def __getitem__(self, idx):
        return self.shuffled_embs[idx], self.target_idx_set[idx]
    
    
def collate_fn_len_sort(batch):
    print(batch)
    # shuffled_embs, target_idx_set = batch
    # shuffled_embs = torch.stack(shuffled_embs, dim=0)
    # target_idx_set = torch.stack(target_idx_set, dim=0)
    # shuffled_embs = [item[0] for item in batch]
    # target_idx_set = [item[1] for item in batch]
    return #shuffled_embs, target_idx_set


class Data_Process:
    def json2dataset(self, args):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        json_list = []
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        sq_set_shuffled = []
        sq_set = []
        target_idx = []
        for item in tqdm(json_list):
            sq = []
            for sq_item in item["sub_questions"]:
                sq+=sq_item
            sq_set.append(sq)
        sq_set = sorted(sq_set, key=len)
        length_sq = [len(item) for item in sq_set]
        bool_list = functional.choose_batch_idx(length_sq)
        sq_set = [sq_set[i] for i in range(len(sq_set)) if bool_list[i]]
        for sq in tqdm(sq_set):
            encoded_sqs = tokenizer(sq, padding=True, truncation=True, max_length=128, return_tensors='pt')
            shuffle_idx = list(range(len(encoded_sqs['input_ids'])))
            np.random.shuffle(shuffle_idx)
            encoded_sqs['input_ids']=encoded_sqs['input_ids'][shuffle_idx]
            encoded_sqs['token_type_ids']=encoded_sqs['token_type_ids'][shuffle_idx]
            encoded_sqs['attention_mask']=encoded_sqs['attention_mask'][shuffle_idx]
            sq_set_shuffled.append(encoded_sqs)

            shuffle_idx = torch.tensor(shuffle_idx)
            perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(shuffle_idx, dim=-1)).transpose(-2, -1).float()
            target_idx.append(perm_ground_truth)

        dataset_size = len(target_idx)
        train_size = int(0.8 * dataset_size / 16) * 16
        val_size = int(0.1 * dataset_size / 16) * 16
        print(dataset_size)
        print(train_size)
        print(val_size)
        dataset = SentenceDataset(sq_set_shuffled, target_idx)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn_len_sort, shuffle=False) #  collate_fn=collate_fn_len_sort,
        val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn_len_sort, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn_len_sort, shuffle=False)
                
        with open("../scratch/data/diff_sort/train_dataloader", "wb") as fp:
            pickle.dump(train_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/val_dataloader", "wb") as fp:
            pickle.dump(val_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/test_dataloader", "wb") as fp:
            pickle.dump(test_dataloader, fp)
        fp.close()

