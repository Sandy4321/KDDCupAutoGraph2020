LOCAL = False
TIME_BUDGET = 1000

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import optim

from torch_geometric.data import Data, DataLoader

from local import read_data

from network import (GCN,
                    ARMA,
                    GGC,
                    SGCN,
                    GraphSAGE,
                    GAT,
                    HOGNN,
                    TAGNN
)

import random

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

fix_seed(1234)

"""
    Model
"""
class Model:
    def __init__(self, features_num = 16, num_class = 2):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.features_num = features_num 
        self.num_class = num_class
        self.data = None
        self.has_weights = True
        self.has_features = True

    """
        Preprocess the data for pytorch geometric
    """
    def generate_pyg_data(self, data):
        x = data['fea_table']
        if x.shape[1] == 1:
            self.has_features = False
            x = x.to_numpy()
            x = x.reshape(x.shape[0])
            x = np.array(pd.get_dummies(x))
        else:
            self.has_features = True
            x = x.drop('node_index', axis = 1).to_numpy()
        x = torch.tensor(x, dtype=torch.float)
        # Edge file
        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key = lambda d: d[0])
        edge_index = torch.tensor(edge_index, dtype = torch.long).transpose(0, 1)
        # Edge weight
        edge_weight = df['edge_weight'].to_numpy()
        self.has_weights = np.unique(edge_weight.flatten()).size > 1 
        edge_weight = torch.tensor(edge_weight, dtype = torch.float32)
        # Y (target)
        num_nodes = x.size(0)
        y = torch.zeros(num_nodes, dtype = torch.long)
        # Y train
        if LOCAL:
            inds = data['train_label'][['node_index']].to_numpy()
            train_y = data['train_label'][['label']].to_numpy()
            y[inds] = torch.tensor(train_y, dtype = torch.long) 
        # Train indices
        train_indices = data['train_indices']
        # Test indices
        test_indices = data['test_indices']
        # Dataset
        data = Data(x = x, edge_index = edge_index, y = y, edge_weight = edge_weight)
        data.num_nodes = num_nodes
        # Train mask
        train_mask = torch.zeros(num_nodes, dtype = torch.bool)
        train_mask[train_indices] = 1
        data.train_mask = train_mask
        # Test mask
        test_mask = torch.zeros(num_nodes, dtype = torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask
        return data

    """
        Update parameters :
         - data
         - features_num
         - num_class
    """
    def update_params(self,data_dict):
        self.data = self.generate_pyg_data(data_dict) 
        self.data = self.data.to(self.device)
        self.features_num = self.data.x.size()[1]
        self.num_class=int(max(self.data.y)+1)

    """
        train the model
    """
    def train(self, model, epochs):
        optimizer = optim.Adam(model.parameters(),lr = 0.01, weight_decay=2e-5)
        cost = 1.0
        for epoch in range(1, epochs+1):
            model.train()
            optimizer.zero_grad()
            loss = F.nll_loss(model(self.data)[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            cost = float(loss.item())
            if LOCAL:
                train_acc, test_acc = self.get_acc(model)
                print("Epoch {} Loss {:2.3f} -- Train_Acc {:2.3f} Test_Acc {:2.3f}".format(
                    epoch, cost, train_acc, test_acc)) 
        return cost
   
    """
        Predict
    """
    @torch.no_grad()
    def predict(self, model):
        model.eval()
        return model(self.data)[self.data.test_mask].max(1)[1]

    """
        Predict from an ensemble using voting strategy
        # TODO add stacking method
    """
    def pred_ensemble(self, method = "voting"):
        if self.has_weights:
            models = [
                (GCN(self.features_num, self.num_class, num_layers = 2, hidden = 128, dropout = 0.2), 200)
                ,(GraphSAGE(self.features_num, self.num_class,num_layers = 2, hidden = 128, dropout = 0.2), 100)
                ,(HOGNN(self.features_num, self.num_class, num_layers = 2, hidden = 16, dropout = 0.2), 200)
                ,(TAGNN(self.features_num, self.num_class, num_layers = 2, hidden = 16, dropout = 0.2), 140)
                ,(SGCN(self.features_num, self.num_class, num_layers = 2, hidden = 16, dropout = 0.2), 200)
            ] 
        if not self.has_weights and self.has_features:
            models = [ 
                (GAT(self.features_num, self.num_class,num_layers = 2, hidden = 128, dropout = 0.2), 140)
                ,(GCN(self.features_num, self.num_class,num_layers = 2, hidden = 128, dropout = 0.2), 200)
                ,(SGCN(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 200)
                ,(ARMA(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 140)
            ]
        if not self.has_weights and not self.has_features:
            models = [
                (GraphSAGE(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 140)
                ,(GAT(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 140) 
                ,(HOGNN(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 100)
                ,(TAGNN(self.features_num, self.num_class,num_layers = 2, hidden = 16, dropout = 0.2), 100)
            ]
        y = []
        cost = []
        for model,epochs in models:
            if LOCAL:
                print("model : ",model)
            model = model.to(self.device)
            c_m = self.train(model, epochs)
            cost.append(c_m)
            y_m = self.predict(model)
            y.append(y_m)
        cost_, y_ = zip(*sorted(zip(cost, y)))
        y_ = list(y_)
        if method == "voting":
            pred, _ = torch.mode(torch.stack(y_[0:3]).t(), 1)
        return pred

    @torch.no_grad()
    def get_acc(self, model):
        model.eval()
        logits = model(self.data)
        acc = []
        for _, mask in self.data('train_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            correct = float(pred.eq(self.data.y[mask]).sum().item())
            a = correct / mask.sum().item()
            acc.append(a)
        return acc[0], acc[1]

    def train_predict(self, data_dict, time_budget): 
        self.update_params(data_dict)
        if LOCAL:
            print("Training the model, this may take a while ...")
        pred = self.pred_ensemble()
        pred = pred.cpu().numpy().flatten()
        if LOCAL:
            predY = torch.tensor(pred, dtype = torch.long)
            test_y = data_dict['test_label'][['label']].to_numpy().flatten()
            trueY = torch.tensor(test_y, dtype = torch.long)
            correct = float (predY.eq(trueY).sum().item())
            acc = correct / self.data.test_mask.sum().item()
            print('Accuracy: {:.3f}'.format(acc))
        return pred

def test_model(num_dataset):
    print("Importing data ...")
    data_dict = read_data(num_dataset)
    print("Data imported")
    model = Model()
    model.train_predict(data_dict, TIME_BUDGET)

if __name__=="__main__":
    LOCAL = True
    num_dataset = str(input())
    test_model(num_dataset)

