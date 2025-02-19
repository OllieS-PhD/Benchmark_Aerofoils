import sys
import h5py
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


def dataLoader(data_path, foil_n, alpha):
    # Got to load in processed data into here
    alf = alpha-4
    dataset_nm = ('foil_' + '{:04d}'.format(foil_n) + '/AoA_'+ f'{alf}')
    G = nx.Graph()
    with h5py.File(data_path, 'r') as hf:
        (cl, cd) = hf[dataset_nm]['coeffs'][()]
        nodes = hf[dataset_nm]['node_attributes'][:]
        for node in tqdm(nodes, desc="Reading Nodes"):
            attrs = dict(nodes[node].attrs)
            G.add_node(int(node), **attrs)
        
        edges = hf[dataset_nm]['edge_attributes'][:]
        for edge in tqdm(edges, desc="Reading Nodes"):
            u = edge.attrs['u']
            v = edge.attrs['v']
            G.add_edge[u,v]
    
    pos = nx.get_node_attributes(G, 'pos')
    node_colours = ['red' if G.nodes[node]['airfoil'] else 'blue' for node in G.nodes()]
    edge_colors = []
    for u, v in tqdm(G.edges(), desc="Processing Edges"):
    # Use the average of the two connected nodes' values as the edge color value
        avg_value = (G.nodes[u]["rho_u"] + G.nodes[v]["rho_u"]) / 2
        edge_colors.append(avg_value)
    # Normalize edge color values for colormap
    edge_colors_normalized = np.array(edge_colors)
    plt.figure()
    print('here')
    nx.draw(G, pos=pos, node_color = node_colours, node_size=10)
    print('here')
    plt.show()


if __name__ == "__main__":
    data_path = 'O:/WindAI_Data/Processed/Airfoil_Processed.h5'
    
    with h5py.File(data_path, 'r') as hf:
        top_groups = [k for k in hf.keys()]
        print(top_groups)
    
    foil_n = 25
    alpha = 0
    dataLoader(data_path, foil_n, alpha)