import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer


class SentenceDataset(Dataset):
    def __init__(self, shuffled_embs, target_embs):
        self.shuffled_embs = shuffled_embs
        self.target_embs = target_embs

    def __len__(self):
        return len(self.target_embs)

    def __getitem__(self, idx):
        return self.shuffled_embs[idx], self.target_embs[idx]
    

class Data_Process:
    def json2dataset(self, args):
        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        encoder = nn.DataParallel(encoder)
        encoder = encoder.to(args.device)
        
        json_list = []
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()

        sq_set = []
        sq_set_shuffled = []
        for item in tqdm(json_list):
            if len(item["sub_questions"][0]) != 8:
                continue
            else:
                sq_embs = torch.from_numpy(encoder.module.encode(item["sub_questions"][0]))
                sq_set.append(sq_embs)
                np.random.shuffle(sq_embs)
                sq_set_shuffled.append(sq_embs)

        dataset_size = len(sq_set)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size

        dataset = SentenceDataset(sq_set_shuffled, sq_set)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = wikiHowNet()
        # model = model.to(device)
        # for ind, item in enumerate(train_dataloader):
        #     if ind == 0:
                
        with open("../scratch/data/diff_sort/train_dataloader", "wb") as fp:
            pickle.dump(train_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/val_dataloader", "wb") as fp:
            pickle.dump(val_dataloader, fp)
        fp.close()
        with open("../scratch/data/diff_sort/test_dataloader", "wb") as fp:
            pickle.dump(test_dataloader, fp)
        fp.close()

