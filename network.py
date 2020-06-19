import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.data import Data

"""
    Graph Convultional Network
"""
from torch_geometric.nn import GCNConv
class GCN(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, num_layers = 2, hidden = 16, dropout = 0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin2 = nn.Linear(hidden, num_class)
        self.first_lin = nn.Linear(features_num, hidden)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.first_lin(x))
        x = F.dropout(x, p = self.dropout, training = self.training)
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__

"""
    Graph SAGE
"""
from torch_geometric.nn import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, num_layers = 2, hidden = 128, dropout = 0.3):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden, hidden))
        self.conv_last = SAGEConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        for conv in self.convs:
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
        x = self.conv_last(x, edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__

"""
    Simple GCN
"""
from torch_geometric.nn import SGConv
class SGCN(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, dropout = 0.3, num_layers = 2, hidden = 16):
        super(SGCN, self).__init__()
        self.conv1 = SGConv(features_num, hidden)
        self.conv2 = SGConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        x = self.conv2(x, edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__

"""
    Graph Attention Network
"""
from torch_geometric.nn import GATConv
class GAT(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, num_layers = 2, hidden = 128, dropout = 0.3):
        super(GAT, self).__init__()
        self.conv1 = GATConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden, hidden))
        self.conv_last = GATConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = F.relu(conv(x, edge_index))
        x = self.conv_last(x, edge_index)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__


"""
    Higher-Order Graph Neural Network
"""
from torch_geometric.nn import GraphConv
class HOGNN(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, num_layers = 2, hidden = 128, dropout = 0.3):
        super(HOGNN, self).__init__()
        self.conv1 = GraphConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GraphConv(hidden, hidden))
        self.conv_last = GraphConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        for conv in self.convs:
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
        x = self.conv_last(x, edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__


"""
 Topology Adaptive Graph Convultional Network
"""
from torch_geometric.nn import GraphConv
class TAGNN(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, num_layers = 2, hidden = 128, dropout = 0.3):
        super(TAGNN, self).__init__()
        self.conv1 = GraphConv(features_num, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GraphConv(hidden, hidden))
        self.conv_last = GraphConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        for conv in self.convs:
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = F.relu(conv(x, edge_index, edge_weight = edge_weight))
        x = self.conv_last(x, edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim = -1)

    def __repr__(self):
        return self.__class__.__name__

"""
    GNN with Convultional ARMA Filters
"""
from torch_geometric.nn import ARMAConv
class ARMA(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, dropout = 0.2, num_layers = 2, hidden = 16):
        super(ARMA, self).__init__()
        self.conv1 = ARMAConv(features_num, hidden)
        self.conv2 = ARMAConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = self.conv1(x, edge_index, edge_weight = edge_weight)
        x = self.conv2(x,edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

"""
    Gated Graph Convultion
"""
from torch_geometric.nn import GatedGraphConv
class GGC(nn.Module):
    def __init__(self, features_num = 16, num_class = 2, dropout = 0.2, num_layers = 2, hidden = 16):
        super(GGC, self).__init__()
        self.conv1 = GatedGraphConv(features_num, hidden)
        self.conv2 = GatedGraphConv(hidden, num_class)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.dropout(x, p = self.dropout, training = self.training)
        x = F.relu(self.conv1(x, edge_index, edge_weight = edge_weight))
        x = self.conv2(x, edge_index, edge_weight = edge_weight)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

