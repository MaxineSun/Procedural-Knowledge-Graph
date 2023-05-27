import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SGConv, APPNP
import torch.nn.functional as F


class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.log_softmax(x)
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
