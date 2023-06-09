import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, APPNP, SAGEConv, GATv2Conv, HeteroConv, GATConv, to_hetero
import torch.nn.functional as F
from torch_geometric.data import HeteroData


class GCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNet, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = self.log_softmax(x)
        return x


class SGCNet(torch.nn.Module):
    def __init__(self, input_dim, out_dim, K=2, cached=True):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(input_dim, out_dim, K=K, cached=cached)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)
    

class APPNPNet(torch.nn.Module):
    def __init__(self, input_dim,  out_dim, num_hid=16, K=1, alpha=0.1, dropout=0.5):
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
# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_user, x_movie, edge_label_index):
        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_channels, data = None):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.movie_lin = torch.nn.Linear(768, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["mq"].num_nodes, hidden_channels)
        self.movie_emb = torch.nn.Embedding(data["sq"].num_nodes, hidden_channels)
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data: HeteroData):
        x_dict = {
          "mq": self.user_emb(data["mq"].node_id),
          "sq": self.movie_lin(data["sq"].x) + self.movie_emb(data["sq"].node_id),
        } 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["mq"],
            x_dict["sq"],
            data["mq","queries","sq"].edge_label_index,
        )
        return pred