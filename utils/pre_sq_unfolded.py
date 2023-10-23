import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, random_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from utils import functional


class Data_Process:
    def json2dataset(self, args):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        sq_set_shuffled = []
        target_idx = []
        batch_id = 0
        for item in tqdm(json_list):
            sq = []
            for sq_item in item["sub_questions"]:
                sq+=sq_item
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
            
            if len(target_idx) == 512:
                with open(f"../scratch/data/diff_sort/sq_set{str(batch_id).zfill(5)}", "wb") as fp:
                    pickle.dump(sq_set_shuffled, fp)
                fp.close()
                with open(f"../scratch/data/diff_sort/target_idx{str(batch_id).zfill(5)}", "wb") as fp:
                    pickle.dump(target_idx, fp)
                fp.close()
                sq_set_shuffled = []
                target_idx = []
                batch_id += 1 