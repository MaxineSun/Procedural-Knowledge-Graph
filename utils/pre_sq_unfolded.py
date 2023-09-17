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
    def __init__(self, shuffled_embs, target_idx_set):
        self.shuffled_embs = shuffled_embs
        self.target_idx_set = target_idx_set

    def __len__(self):
        return len(self.target_idx_set)

    def __getitem__(self, idx):
        return self.shuffled_embs[idx], self.target_idx_set[idx]
    
    
def collate_fn_len_sort(batch):
    batch_tensor = []
    for shuffled_embs, target_idx_set in zip(*batch):
        batch_tensor.append([shuffled_embs, target_idx_set])
    return batch_tensor


class Data_Process:
    def json2dataset(self, args):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        json_list = json_list[:1000]
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
        for id_, sq in enumerate(tqdm(sq_set)):
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

        print(sq_set_shuffled[67])
        
        dataset_size = len(target_idx)
        train_size = int(0.8 * dataset_size / 16) * 16
        val_size = int(0.1 * dataset_size / 16) * 16

        dataset = SentenceDataset(sq_set_shuffled, target_idx)
        train_dataset = dataset[:train_size]
        val_dataset = dataset[train_size:train_size + val_size]
        test_dataset = dataset[train_size + val_size:]
        
        train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn_len_sort)
        val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=collate_fn_len_sort)
        test_dataloader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn_len_sort)
                
        with open("../scratch/data/diff_sort/train_dataloader16", "wb") as fp:
            pickle.dump(train_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/val_dataloader16", "wb") as fp:
            pickle.dump(val_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/test_dataloader16", "wb") as fp:
            pickle.dump(test_dataloader, fp)
        fp.close()