import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class SentenceDataset(Dataset):
    def __init__(self,shuffled_embs, target_idx_set):
        self.shuffled_embs = shuffled_embs
        self.target_idx_set = target_idx_set

    def __len__(self):
        return len(self.target_idx_set)

    def __getitem__(self, idx):
        return self.shuffled_embs[idx], self.target_idx_set[idx]
    

class Data_Process:
    def json2dataset(self, args):
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        json_list = []
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()
        # mqs = [item["main_question"] for item in json_list]
        # encoded_mqs = tokenizer(mqs, padding=True, truncation=True, max_length=128, return_tensors='pt')
        # sq_set = []
        sq_set_shuffled = []
        target_idx = []
        for item in tqdm(json_list):
            sq = []
            for sq_item in item["sub_questions"]:
                sq+=sq_item
            encoded_sqs = tokenizer(sq, padding=True, truncation=True, max_length=128, return_tensors='pt')
            # sq_set.append(encoded_sqs)
            shuffle_idx = list(range(len(encoded_sqs['input_ids'])))
            np.random.shuffle(shuffle_idx)
            encoded_sqs['input_ids']=encoded_sqs['input_ids'][shuffle_idx]
            encoded_sqs['token_type_ids']=encoded_sqs['token_type_ids'][shuffle_idx]
            encoded_sqs['attention_mask']=encoded_sqs['attention_mask'][shuffle_idx]
            sq_set_shuffled.append(encoded_sqs)

            shuffle_idx = torch.tensor(shuffle_idx)
            perm_ground_truth = torch.nn.functional.one_hot(torch.argsort(shuffle_idx, dim=-1)).transpose(-2, -1).float()
            target_idx.append(perm_ground_truth)

        # with open("../scratch/data/diff_sort/encoded_mqs", "wb") as fp:
        #     pickle.dump(encoded_mqs, fp)
        # fp.close()
        # with open("../scratch/data/diff_sort/sq_set", "wb") as fp:
        #     pickle.dump(sq_set_shuffled, fp)
        # fp.close()
        with open("../scratch/data/diff_sort/sq_set_shuffled", "wb") as fp:
            pickle.dump(sq_set_shuffled, fp)
        fp.close()

        with open("../scratch/data/diff_sort/target_idx", "wb") as fp:
            pickle.dump(target_idx, fp)
        fp.close()

        # with open("../scratch/data/diff_sort/encoded_mqs", "rb") as fp:
        #     encoded_mqs = pickle.load(fp)
        # fp.close()
        # with open("../scratch/data/diff_sort/sq_set_shuffled", "rb") as fp:
        #     sq_set_shuffled = pickle.load(fp)
        # fp.close()
        # with open("../scratch/data/diff_sort/target_idx", "rb") as fp:
        #     target_idx = pickle.load(fp)
        # fp.close()

        # dataset_size = len(sq_set)
        # train_size = int(0.8 * dataset_size)
        # val_size = int(0.1 * dataset_size)
        # test_size = dataset_size - train_size - val_size

        # dataset = SentenceDataset(mq_set, sq_set_shuffled, sq_set)
        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        # val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
        # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
                
        # with open("../scratch/data/diff_sort/train_dataloader", "wb") as fp:
        #     pickle.dump(train_dataloader, fp)
        # fp.close()
        # with open("../scratch/data/diff_sort/val_dataloader", "wb") as fp:
        #     pickle.dump(val_dataloader, fp)
        # fp.close()
        # with open("../scratch/data/diff_sort/test_dataloader", "wb") as fp:
        #     pickle.dump(test_dataloader, fp)
        # fp.close()

