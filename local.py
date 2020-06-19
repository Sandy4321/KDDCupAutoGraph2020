import numpy as np
import pandas as pd

#Importing dataset
def read_data(num_dataset):
    filepath = 'public/' + num_dataset
    test_label = pd.read_csv(filepath + '/test_label.tsv', sep = '\t')
    edge = pd.read_csv(filepath + '/train.data/edge.tsv', sep = '\t')
    feature = pd.read_csv(filepath + '/train.data/feature.tsv', sep = '\t')
    test_node_id = pd.read_csv(filepath + '/train.data/test_node_id.txt',sep = ' ', header = None)
    train_node_id = pd.read_csv(filepath + '/train.data/train_node_id.txt',sep = ' ', header = None)
    train_label = pd.read_csv(filepath + '/train.data/train_label.tsv', sep='\t')
    data = {}
    data['fea_table'] = feature
    data['edge_file'] = edge
    data['train_indices'] = train_node_id.iloc[:, 0].tolist()
    data['test_indices'] = test_node_id.iloc[:, 0].tolist()
    data['train_label'] = train_label
    data['test_label'] = test_label
    return data

