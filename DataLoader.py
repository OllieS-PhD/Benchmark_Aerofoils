import sys
import h5py
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


def dataLoader(foil_n, alpha):
    # Got to load in processed data into here
    alf = alpha-4
    alf_path = f'AoA_{alf}'
    vars = ["rho","rho_u","rho_v", "e", "omega", "airfoil"]#,"pos"]
    data_path = 'E:/turb_model/Re_3M/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
    G = nx.Graph()
    with h5py.File(data_path, 'r') as hf:
        mesh_sz = hf[alf_path]['nodes']['rho'][()].size
        data = np.empty((len(vars), mesh_sz))
        (cl, cd) = hf[alf_path]['coeffs'][()]
        
        #nodes
        pos = hf[alf_path]['nodes']['pos'][()]
        for i in range(len(vars)):
            temp = hf[alf_path]['nodes'][vars[i]][()]
            data[i,:] = hf[alf_path]['nodes'][vars[i]][:][()]
        
        for i in range(mesh_sz): #tqdm(range(mesh_sz), desc="Adding Nodes"):
            G.add_node(i, pos=pos, rho=data[0,i], rho_u=data[1,i], rho_v=data[2,i], e=data[3,i], omega=data[4,i], airfoil=data[5,i])
        
        #edges
        edge_arr = hf[alf_path]['edges'][()]
        for edge in edge_arr:
            G.add_edge(edge[0], edge[1])
    
    
    G_init = nx.Graph()
    G_init.add_nodes_from(G)
    G_init.add_edges_from(G.edges())
    Ma_inf = 0.1
    rho_inf = 1
    rho_u_inf = Ma_inf * math.cos(alf)
    rho_v_inf = Ma_inf * math.sin(alf)
    c = 340.15
    vel_inf = Ma_inf * c
    u_inf = vel_inf * math.cos(alf)
    v_inf = vel_inf * math.sin(alf)
    for node in G_init.nodes():
        G_init.nodes[node]['rho'] = rho_inf
        G_init.nodes[node]['rho_u'] = 0#rho_u_inf
        G_init.nodes[node]['rho_v'] = 0#rho_v_inf
        G_init.nodes[node]['e'] = 0
        G_init.nodes[node]['omega'] = 0
    
    
    
    # node_colours = ['red' if G.nodes[node]['airfoil'] else 'blue' for node in G.nodes()]
    # plt.figure()
    # print('Making Graph')
    # nx.draw(G, pos=pos, node_color = node_colours, node_size=10)
    # plt.show()
    
    return G_init, G, cl, cd


if __name__ == "__main__":
    data_path = 'E:/turb_model/Re_3M/Airfoil_0000.h5'
    
    with h5py.File(data_path, 'r') as hf:
        top_groups = [k for k in hf["AoA_0"]['nodes'].keys()]
        print(top_groups)
    
    foil_n = 25
    alpha = 0
    dataLoader(foil_n, alpha)