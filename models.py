import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv,
    SGConv,
    APPNP,
    SAGEConv,
    GATv2Conv,
    HeteroConv,
    GATConv,
    to_hetero,
)
import torch.nn.functional as F
import torch.nn.init as init
import utils.parse_args as pa
from transformers import AutoModel
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
from utils import diffsort, functional


class GCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.softmax(x)
        return x


class SGCNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim, K=2, cached=True):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(input_dim, out_dim, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class APPNPNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim, num_hid=16, K=1, alpha=0.1, dropout=0.5):
        super(APPNPNet, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, num_hid)
        self.lin2 = torch.nn.Linear(num_hid, out_dim)
        self.prop1 = APPNP(K, alpha)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv1 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv2 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv3 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv4 = GATv2Conv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv3 = GATConv(hidden_channels, hidden_channels, add_self_loops = False)
        # self.conv4 = GATConv(hidden_channels, hidden_channels, add_self_loops = False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = self.conv4(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, mq, sq, edge_label_index):
        edge_feat_user = mq[edge_label_index[0]]
        edge_feat_movie = sq[edge_label_index[1]]
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)


class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels, data=None):
        super().__init__()
        self.lin = torch.nn.Linear(768, hidden_channels)
        self.mq_emb = torch.nn.Embedding(data["mq"].num_nodes, hidden_channels)
        self.sq_emb = torch.nn.Embedding(data["sq"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
            "mq": self.lin(data["mq"].x), 
            "sq": self.lin(data["sq"].x), 
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["mq"],
            x_dict["sq"],
            data["mq", "queries", "sq"].edge_label_index,
        )
        return pred


class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, data=None):
        super().__init__()
        self.lin1 = torch.nn.Linear(768, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.mq_emb = torch.nn.Embedding(data["mq"].num_nodes, hidden_channels)
        self.sq_emb = torch.nn.Embedding(data["sq"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData):
        x_dict = {
            "mq": self.lin2(torch.relu(self.lin1(data["mq"].x))),
            "sq": self.lin2(torch.relu(self.lin1(data["sq"].x)))
        }

        pred = self.classifier(
            x_dict["mq"],
            x_dict["sq"],
            data["mq", "queries", "sq"].edge_label_index,
        )
        return pred


class wikiHowNet(nn.Module):
    def __init__(self):
        super(wikiHowNet, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        init.normal_(self.fc1.weight, mean=1.0, std=0.30)
        self.fc2 = nn.Linear(128, 1)
        init.normal_(self.fc2.weight, mean=1.0, std=0.30)
        # self.fc3 = nn.Linear(128, 1)
        # init.normal_(self.fc2.weight, mean=1.0, std=0.30)
        self.act1 = torch.nn.LeakyReLU()
        # self.act2 = torch.nn.LeakyReLU()
        self.encode_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def forward(self, x):
        args = pa.parse_args()
        encode_shuffled_sq = self.encode_model(**x)
        sq_shuffled_emb = functional.mean_pooling(encode_shuffled_sq, x['attention_mask'])
        len_sq = len(sq_shuffled_emb)
        sq_shuffled_emb = self.fc1(sq_shuffled_emb)
        sq_shuffled_emb = self.act1(sq_shuffled_emb)
        shuffle_scalars = self.fc2(sq_shuffled_emb)
        # sq_shuffled_emb = self.act2(sq_shuffled_emb)
        # shuffle_scalars = self.fc3(sq_shuffled_emb)
        shuffle_scalars = shuffle_scalars.view(1, -1)
        sorter = diffsort.DiffSortNet(
            sorting_network_type='odd_even',
            size=len_sq,
            device=args.device,
            steepness=args.steepness,
            art_lambda=args.art_lambda,
        )
        _, perm_prediction = sorter(shuffle_scalars)
        perm_prediction = perm_prediction.squeeze(0)
        return shuffle_scalars, perm_prediction