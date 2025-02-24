from DataLoader import dataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import TopKPooling, knn_interpolate
import torch_geometric.utils as pyg_utils

import numpy as np
import networkx as nx

class GNN(nn.Module):
    def __init__(self, in_dim):
        super(GNN, self).__init__()
        self.convs = nn.ModuleList()
        


'''
Need classes for 
- Convolutional Blocks
- Pooling layers
- Unpooling layers
'''