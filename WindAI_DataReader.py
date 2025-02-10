# Imports
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from scipy.stats import gaussian_kde
from scipy.spatial import Delaunay
import time
import math
import rounders
import tqdm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx
from shapely.geometry import Polygon

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
# from torch_geometric import Data

# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
alt.data_transformers.disable_max_rows()




'''
For dataLoader:
https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html
https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/loader/neighbor_loader.html
https://github.com/jordan7186/Edgeless-GNN-external/blob/main/utils.py#L646
https://arxiv.org/pdf/2104.05225
'''
def dataLoader(data_path, model, Re, foil_n, alpha):
    var = ["x","y","rho","rho_u","rho_v"]
    var_sz = len(var)
    G = nx.Graph()
    #print(range(len(dtype)))
    tic = time.time()
    with h5py.File(data_path, 'r') as hf:
        lm = hf['shape']['landmarks'][foil_n][()]
        mesh_sz = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)]['x'][:,alpha][()].size
        data = np.empty((var_sz, mesh_sz))
        for i in range(var_sz):
            data[i,:] = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)][var[i]][:,alpha][()]
        #data = torch.Tensor(np.swapaxes(data, 0, 1))
        
        cl = hf[model][Re]['C_l'][foil_n,alpha][()]
        cd = hf[model][Re]['C_d'][foil_n,alpha][()]
    
    xk, yk = data[0,:], data[1,:]
    X1 = np.vstack((xk,yk)).T
    tri = Delaunay(X1)
    
    for i in range(mesh_sz):
        G.add_node(i, pos=X1[i], rho=data[2,i], rho_u=data[3,i], rho_v=data[4,i])
    
    # Check each triangle and add edges only if it doesn't intersect the exclusion area
    for simplex in tri.simplices:
        triangle = [X1[simplex[0]], X1[simplex[1]], X1[simplex[2]]]
        triangle_polygon = Polygon(triangle)
        
        # Skip triangles that intersect the exclusion polygon
        if not Polygon(lm).intersects(triangle_polygon):
            G.add_edge(simplex[0], simplex[1])
            G.add_edge(simplex[1], simplex[2])
            G.add_edge(simplex[2], simplex[0])
    print('#edges:  ', G.number_of_edges())
    print('#nodes:  ',G.number_of_nodes())
    tok = time.time()
    pos = nx.get_node_attributes(G, 'pos')
    #print(pos)
    nx.draw(G, pos, with_labels=False, node_size=0.1)
    plt.plot(lm[:,0], lm[:,1], color='red')
    plt.show()
    print('\n\nElapsed Time to Read 1 AoA: ', tok-tic,' s')
    print(data.shape)
    print(data.dtype)
    print('cl:  ', cl)
    print('cd:  ', cd)
    return G, cl, cd









if __name__ == '__main__':
    # Set data path here
    data_path = 'O:/WindAI_Data/2k/airfoil_2k_data.h5'
    dataLoader(data_path=data_path, model='trans_model', Re='Re03000000', foil_n=0, alpha=0)