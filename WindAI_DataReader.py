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
from tqdm import tqdm
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import networkx as nx
from shapely.geometry import Polygon, Point
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
def dataSorter(data_path_load, data_path_save, model, Re, foil_n, alpha):
    var = ["x","y","rho","rho_u","rho_v", "e"]
    var_sz = len(var)
    G = nx.Graph()
    #print(range(len(dtype)))
    # tic = time.time()
    with h5py.File(data_path_load, 'r') as hf:
        lm = hf['shape']['landmarks'][foil_n][()]
        mesh_sz = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)]['x'][:,alpha][()].size
        data = np.empty((var_sz, mesh_sz))
        for i in range(var_sz):
            data[i,:] = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)][var[i]][:,alpha][()]
        #data = torch.Tensor(np.swapaxes(data, 0, 1))
        
        alf = hf[model][Re]['alpha'][alpha][()]
        cl = hf[model][Re]['C_l'][foil_n,alpha][()]
        cd = hf[model][Re]['C_d'][foil_n,alpha][()]


    
    xk, yk = data[0,:], data[1,:]
    X1 = np.vstack((xk,yk)).T
    tri = Delaunay(X1)
    # plt.figure()
    # plt.plot(xk,yk,'.')
    # plt.plot(lm[:,0], lm[:,1], '-')
    # plt.show()
    
    for i in range(mesh_sz): #tqdm(range(mesh_sz), desc="Adding Nodes"):
        G.add_node(i, pos=X1[i], rho=data[2,i], rho_u=data[3,i], rho_v=data[4,i], e=data[5,i], airfoil = False)
    # for i in range(len(lm)):
    #     G.add_node(mesh_sz+i, rho = )
    pos = nx.get_node_attributes(G, 'pos')
    
    aero_poly = Polygon(lm)
        # Check each triangle and add edges only if it doesn't intersect the exclusion area
    for simplex in tri.simplices: #tqdm(tri.simplices, desc="Adding Edges"):
        triangle = [X1[simplex[0]], X1[simplex[1]], X1[simplex[2]]]
        triangle_polygon = Polygon(triangle)
        # Skip triangles that intersect the exclusion polygon
        if not aero_poly.intersection(triangle_polygon):
            if not G.has_edge(simplex[0], simplex[1]): G.add_edge(simplex[0], simplex[1])
            if not G.has_edge(simplex[1], simplex[2]): G.add_edge(simplex[1], simplex[2])
            if not G.has_edge(simplex[2], simplex[0]): G.add_edge(simplex[2], simplex[0])
    buff_dist = 3.5e-6
    buff_poly = aero_poly.buffer(buff_dist)
    n_foil_points = 0
    for nid in G.nodes():
        if buff_poly.contains(Point(G.nodes[nid]['pos'])):
            G.nodes[nid]['airfoil'] = True
            n_foil_points +=1
    # print(n_foil_points)
    
    #(aero_id, aero_pos) = [(nid, G.nodes[nid]['pos']) for nid in G.nodes() if G.nodes[nid]['airfoil']]
    aero_ids = []
    aero_pos = []
    for nid in G.nodes():
        if G.nodes[nid]['airfoil']:
            aero_ids.append(nid)
            aero_pos.append(G.nodes[nid]['pos'])  
    if n_foil_points==len(aero_ids):
        for i in range(len(aero_pos)):
            # for nid in G.nodes():
            #     if G.nodes[nid]['airfoil']:
            #         curr_nid = nid
            #         break
            curr_nid = aero_ids[i]
            if i == 0:
                prev_nid = curr_nid
                nid0 = curr_nid
            else:
                pos_u = G.nodes[curr_nid]['pos']
                pos_v = G.nodes[nid0 if i ==len(aero_pos) else prev_nid]['pos']
                edge_len = math.sqrt((pos_u[0] - pos_v[0])**2 + (pos_u[1] - pos_v[1])**2)
                if i == len(aero_pos):
                    if edge_len<0.1 and not G.has_edge(curr_nid, nid0) : G.add_edge(curr_nid, nid0)
                else:
                    if edge_len<0.1 and not G.has_edge(prev_nid, curr_nid) : G.add_edge(prev_nid, curr_nid)
            prev_nid = curr_nid
    
    
    #####################################################
    #                                                   #
    #                 Save to Processed                 #
    #                                                   #
    #####################################################
    adj_matrix = nx.to_numpy_array(G)
    node_attrs = {node: dict(G.nodes[node]) for node in G.nodes()}
    edge_attrs = {(u, v): dict(G.edges[u, v]) for u, v in G.edges()}
    
    dataset_nm = ('foil_' + '{:04d}'.format(foil_n) + '_AoA_'+ f'{alf}')
    with h5py.File(data_path_save, 'w') as hf:
        grp = hf.create_group(dataset_nm)
        grp.create_dataset('coeffs', data = (cl,cd))
        
        node_group = grp.create_group('node_attributes')
        for node, attrs in tqdm(node_attrs.items(), desc="Saving Nodes"):
            node_groupie = node_group.create_group(str(node))
            for key, value in attrs.items():
                node_groupie.attrs[key] = value
        
        edge_group = grp.create_group('edge_attributes')
        idx = -1
        for edge, _ in tqdm(edge_attrs.items(), desc="Saving Edges"):
            idx+=1
            edge_groupie = edge_group.create_group(str(idx))#, data=np.array(attrs))
            edge_groupie.attrs['u'] = edge[0]
            edge_groupie.attrs['v'] = edge[1]
    
    
    
    
    
    
    
    
    
    

    # tok = time.time()

    # node_colours = ['red' if G.nodes[node]['airfoil'] else 'blue' for node in G.nodes()]
    # edge_colors = []
    # for u, v in tqdm(G.edges(), desc="Processing Edges"):
    # # Use the average of the two connected nodes' values as the edge color value
    #     avg_value = (G.nodes[u]["rho_u"] + G.nodes[v]["rho_u"]) / 2
    #     edge_colors.append(avg_value)
    # # Normalize edge color values for colormap
    # edge_colors_normalized = np.array(edge_colors)
    # plt.figure()
    # print('here')
    # nx.draw(G, pos=pos, node_color = node_colours, node_size=10)
    # print('here')
    # plt.show()
    # print('here')
    # node_list = []
    # bc_pos = []
    # bc_rho = []
    # bc_u = []
    # bc_v = []
    # for n in tqdm(G.nodes(), desc="Processing Nodes"):
    #     # if [G.nodes[n]['pos'][0], G.nodes[n]['pos'][1]] in lm:
    #     if abs(math.sqrt((G.nodes[n]['pos'][0])**2 + (G.nodes[n]['pos'][1])**2) - 1) <=1:
    #         #print(G.nodes[n]['rho'],'       ',G.nodes[n]['rho_u'],'       ',G.nodes[n]['rho_v'],'       ',G.nodes[n]['e'])
    #         bc_rho.append(G.nodes[n]['rho'])
    #         bc_u.append(G.nodes[n]['rho_u'])
    #         bc_v.append(G.nodes[n]['rho_v'])
    #         bc_pos.append(G.nodes[n]['pos'])

    
    # ic = []
    # ic.append(sum(bc_rho)/len(bc_rho))
    # ic.append(sum(bc_u)/len(bc_u))
    # ic.append(sum(bc_v)/len(bc_v))
    # print(ic)

    # G_init = G.copy()
    # G_init.nodes[:]['rho'] = ic[0]
    # G_init.nodes[:]['rho_u'] = ic[1]
    # G_init.nodes[:]['rho_v'] = ic[2]
    
    
    # print('\n\nElapsed Time to Read 1 AoA: ', tok-tic,' s')
    # print('#edges:  ', G.number_of_edges())
    # print('#nodes:  ',G.number_of_nodes())
    # print(data.shape)
    # print(data.dtype)
    # print('alf: ', alf)
    # print('cl:  ', cl)
    # print('cd:  ', cd)
    #return G, cl, cd









if __name__ == '__main__':
    # Set data path here
    data_path_raw = 'O:/WindAI_Data/raw/airfoil_2k_data.h5'
    data_path_proc = 'O:/WindAI_Data/Processed/Airfoil_Processed.h5'
    for foil_i in tqdm(range(1830), desc="Foil Number"):
        for alph_i in tqdm(range(24), desc="Alpha"):
            print('\n\n')
            print("###################################################################")
            print(f"                     Foil Number {foil_i}/1830                               ")
            print(f"                     Angle of Attack {alph_i-4}                               ")
            print("###################################################################")
            dataSorter(data_path_load=data_path_raw, data_path_save=data_path_proc, model='turb_model', Re='Re03000000', foil_n=foil_i, alpha=alph_i)
    