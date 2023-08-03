import jsonlines
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def shift_list(lst, a):
    n = len(lst)
    a = a % n  
    shifted_lst = lst[a:] + lst[:a]
    return shifted_lst


class SentenceDataset(Dataset):
    def __init__(self, input_ids, target_ids, y):
        self.input_ids = input_ids
        self.target_ids = target_ids
        self.y = y

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx], self.y[idx]
    

class Data_Process:
    def json2dataset(self, args):
        json_list = []
        # with open(args.jsonpath, "r+") as f_in:
        #     for json in jsonlines.Reader(f_in):
        #         if len(json_list) < 96:
        #             json_list.append(json)
        with open("../scratch/data/json_list", "rb") as fp:
            json_list = pickle.load(fp)
        fp.close()

        input_head = [item["main_question"] for item in json_list]
        target_head = [item["sub_questions"][0][0] for item in json_list]
        y_head = [1]*len(input_head)
        y_head += [0]*len(input_head)
        input_head = input_head*2
        target_head = target_head + shift_list(target_head, 4)

        input_follow = []
        target_follow = []
        for item in json_list:
            for ind, sq in enumerate(item["sub_questions"][0][:-1]):
                input_follow.append(sq)
                next_item = item["sub_questions"][0][ind+1]
                target_follow.append(next_item)

        y_follow = [1]*len(input_follow)
        y_follow += [0]*len(input_follow)
        input_follow = input_follow*2
        target_follow = target_follow + shift_list(target_follow, 4)

        input = input_head + input_follow
        target = target_head + target_follow
        y = y_head + y_follow

        dataset_size = len(input)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size

        dataset_head = SentenceDataset(input, target, y)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset_head, [train_size, val_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=args.nsg_batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.nsg_batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=args.nsg_batch_size, shuffle=True)

        with open("../scratch/data/nsg/train_dataloader", "wb") as fp:
            pickle.dump(train_dataloader, fp)
        fp.close()
        with open("../scratch/data/nsg/val_dataloader", "wb") as fp:
            pickle.dump(val_dataloader, fp)
        fp.close()
        with open("../scratch/data/nsg/test_dataloader", "wb") as fp:
            pickle.dump(test_dataloader, fp)
        fp.close()
