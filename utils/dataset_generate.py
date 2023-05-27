import torch
from torch_geometric.data import InMemoryDataset, Data


class PKG_Dataset(InMemoryDataset):
    def __init__(self, root, x, edge_index, y, transform=None, pre_transform=None):
        self.x =x
        self.edge_index = edge_index
        self.y=y
        super(PKG_Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            "wiki_how_dataset.pt",
        ]

    @property
    def processed_file_names(self):
        return [
            "wiki_how_dataset.pt",
        ]

    def download(self):
        pass

    def process(self):
        graph = Data(x=self.x, edge_index=self.edge_index.t().contiguous(), y=self.y)

        if self.pre_filter is not None:
            graph = [data for data in graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph = [self.pre_transform(data) for data in graph]

        data, slices = self.collate([graph])
        torch.save((data, slices), self.processed_paths[0])
