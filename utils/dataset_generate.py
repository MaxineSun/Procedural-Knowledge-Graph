import torch
import pickle
import numpy as np
import os.path as osp
from torch_geometric.data import Dataset, Data


class PKG_Dataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PKG_Dataset, self).__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["dataset_light.pt"]

    def download(self):
        pass

    def index_to_mask(self, index, size):
        mask = torch.zeros(size, dtype=torch.bool)
        mask[index] = 1
        return mask


    def random_splits(self, size, train_len, val_len, seed=42):
        index = [i for i in range(0, size)]
        train_idx = []
        rnd_state = np.random.RandomState(seed)
        train_len = int(round(0.6 * size))
        train_idx = rnd_state.choice(index, train_len, replace=False)
        rest_index = [i for i in index if i not in train_idx]
        val_idx = rnd_state.choice(rest_index, val_len, replace=False)
        test_idx = [i for i in rest_index if i not in val_idx]

        train_mask = self.index_to_mask(train_idx, size)
        val_mask = self.index_to_mask(val_idx, size)
        test_mask = self.index_to_mask(test_idx, size)
        return train_mask, val_mask, test_mask

    def process(self):
        with open("../scratch/data/x", "rb") as fp:
            x = pickle.load(fp)
        fp.close()
        with open("../scratch/data/ed", "rb") as fp:
            edge_list = pickle.load(fp)
        fp.close()
        with open("../scratch/data/y", "rb") as fp:
            y = pickle.load(fp)
        fp.close()

        graph = Data(
            x=torch.from_numpy(x),
            edge_index=torch.from_numpy(edge_list),
            y=torch.from_numpy(y),
        )
        train_len = int(round(0.6 * len(graph.y)))
        val_len = int(round(0.2 * len(graph.y)))
        train_mask, val_mask, test_mask = self.random_splits(len(graph.y), train_len, val_len)

        graph.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        graph.val_mask = torch.tensor(val_mask, dtype=torch.bool)
        graph.test_mask = torch.tensor(test_mask, dtype=torch.bool)

        if self.pre_filter is not None:
            graph = [data for data in graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph = [self.pre_transform(data) for data in graph]

        torch.save(graph, self.processed_paths[0])

    def len(self):
        return len(self.processed_file_names)

    def get(self):
        data = torch.load(osp.join(self.processed_dir, "dataset_light.pt"))
        return data
