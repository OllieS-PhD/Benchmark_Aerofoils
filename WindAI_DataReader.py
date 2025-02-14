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
import tqdm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx
from shapely.geometry import Polygon
import pyvista as pv

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
    var = ["x","y","rho","rho_u","rho_v", "e"]
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
        
        alf = hf[model][Re]['alpha'][alpha][()]
        cl = hf[model][Re]['C_l'][foil_n,alpha][()]
        cd = hf[model][Re]['C_d'][foil_n,alpha][()]
    # print(lm)
    
    
    xk, yk = data[0,:], data[1,:]
    X1 = np.vstack((xk,yk)).T
    tri = Delaunay(X1)
    
    for i in range(mesh_sz):
        G.add_node(i, pos=X1[i], rho=data[2,i], rho_u=data[3,i], rho_v=data[4,i], e=data[5,i])
    
    # Check each triangle and add edges only if it doesn't intersect the exclusion area
    for simplex in tri.simplices:
        triangle = [X1[simplex[0]], X1[simplex[1]], X1[simplex[2]]]
        triangle_polygon = Polygon(triangle)
        
        # Skip triangles that intersect the exclusion polygon
        if not Polygon(lm).intersects(triangle_polygon):
            G.add_edge(simplex[0], simplex[1])
            G.add_edge(simplex[1], simplex[2])
            G.add_edge(simplex[2], simplex[0])
    tok = time.time()

    #print(pos)
    # edge_colors = []
    # for u, v in G.edges():
    # # Use the average of the two connected nodes' values as the edge color value
    #     avg_value = (G.nodes[u]["rho_u"] + G.nodes[v]["rho_u"]) / 2
    #     edge_colors.append(avg_value)
    # # Normalize edge color values for colormap
    # edge_colors_normalized = np.array(edge_colors)
    # plt.figure()
    # nx.draw(G, pos, edge_color=edge_colors_normalized, with_labels=False, node_size=0.10)
    # plt.show()
    node_list = []
    bc_pos = []
    bc_rho = []
    bc_u = []
    bc_v = []
    for n in G.nodes():
        # if [G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]] in lm:
        if abs(math.sqrt((G.nodes[n]['pos'][0])**2 + (G.nodes[n]['pos'][1])**2) - 1) <=1:
            print(G.nodes[n]['rho'],'       ',G.nodes[n]['rho_u'],'       ',G.nodes[n]['rho_v'],'       ',G.nodes[n]['e'])
            bc_rho.append(G.nodes[n]['rho'])
            bc_u.append(G.nodes[n]['rho_u'])
            bc_v.append(G.nodes[n]['rho_v'])
            bc_pos.append(G.nodes[n]['pos'])

    
    ic = []
    ic.append(sum(bc_rho)/len(bc_rho))
    ic.append(sum(bc_u)/len(bc_u))
    ic.append(sum(bc_v)/len(bc_v))
    print(ic)

    G_init = G.copy()
    G_init.nodes[:]['rho'] = ic[0]
    G_init.nodes[:]['rho_u'] = ic[1]
    G_init.nodes[:]['rho_v'] = ic[2]
    
    '''
    Here G_init needs to be defined so that all of the nodes are set to be the initial conditions
    uniform density and momentum across the board
    More importantly, need to work out what it all actually means
    '''
    
    
    print('\n\nElapsed Time to Read 1 AoA: ', tok-tic,' s')
    print('#edges:  ', G.number_of_edges())
    print('#nodes:  ',G.number_of_nodes())
    print(data.shape)
    print(data.dtype)
    print('alf: ', alf)
    print('cl:  ', cl)
    print('cd:  ', cd)
    return G_init, G, cl, cd









if __name__ == '__main__':
    # Set data path here
    data_path = 'O:/WindAI_Data/2k/airfoil_2k_data.h5'
    dataLoader(data_path=data_path, model='turb_model', Re='Re03000000', foil_n=0, alpha=24)