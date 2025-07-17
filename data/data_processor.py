# Imports
import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import time
import math
from tqdm import tqdm

import multiprocessing

import networkx as nx
from shapely import distance
from shapely.geometry import Polygon, Point
import pyvista as pv


# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
alt.data_transformers.disable_max_rows()

def dataSorter(foil_n, alpha):
    data_path_load = 'O:/WindAI_Data/raw/airfoil_2k_data.h5'
    data_path_save = 'E:/turb_model/Re_3M/Airfoil_'+'{:04d}'.format(foil_n)+'.h5'
    model='turb_model'
    Re='Re03000000'
    var = ["x","y","rho","rho_u","rho_v", "e", "omega"]
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
    for i in range(mesh_sz):        #tqdm(range(mesh_sz), desc="Adding Nodes"):
        G.add_node(i, x=xk[i], y=yk[i], rho=data[2,i], rho_u=data[3,i], rho_v=data[4,i], e=data[5,i], omega=data[6,i], airfoil = False, dist = 0)
    # for i in range(len(lm)):
    #     G.add_node(mesh_sz+i, rho = )
    pos = (nx.get_node_attributes(G, 'x'), nx.get_node_attributes(G, 'y')) 

    aero_poly = Polygon(lm)
    
    buff_dist = 3.5e-6
    buff_poly = aero_poly.buffer(buff_dist)
    n_foil_points = 0
    for nid in G.nodes():
        point = Point( (G.nodes[nid]['x'], G.nodes[nid]['y']) )
        G.nodes[nid]['dist'] = distance(aero_poly, point)
        if buff_poly.contains( point ):
            G.nodes[nid]['airfoil'] = True
            n_foil_points +=1
    
    att_vars = dict(G.nodes[0]).keys()
    # print(att_vars)
    
    print_graph = False
    if print_graph:
        node_colours = ['red' if G.nodes[node]['airfoil'] else 'blue' for node in G.nodes()]
        edge_colors = []
        for u, v in G.edges():   #tqdm(G.edges(), desc="Processing Edges"):
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
    
    alf_path = ('AoA_'+ f'{alf}')
    with h5py.File(data_path_save, 'a') as hf:
        # grp_foil = hf.create_group(foil_path)
        grp = hf.create_group(alf_path)
        grp.create_dataset('coeffs', data = (cl,cd))
        # grp.create_dataset('edges', data=edge_arr)#, compression = "gzip")
        
        lm_grp = grp.create_group('lm')
        lm_grp.create_dataset('x', data=lm[:,0])
        lm_grp.create_dataset('y', data=lm[:,1])
        # Loop through each attribute
        node_group = grp.create_group('nodes')
        for att in att_vars:        #tqdm(att_vars, desc='Creating Datasets'):
            # data = []
            # for nid in G.nodes():
            #     data.append(G.nodes[nid][att])
            att_data = [G.nodes[nid][att] for nid in G.nodes()]
            node_group.create_dataset(att, data=att_data)#, compression = "gzip")
    
    # tok = time.time()


def worker(worker_num):
    n_batch = 122
    start = 0 + ((n_batch)* worker_num)
    end = start + n_batch
    
    if worker_num == 0:
        for foil_i in tqdm(range(start, end), desc='Worker 0 Progress'):
            for alph_i in range(25):
                dataSorter(foil_n=foil_i, alpha=alph_i)
    else:    
        for foil_i in range(start, end):
            for alph_i in range(25):
                dataSorter(foil_n=foil_i, alpha=alph_i)


if __name__ == '__main__':
    tik = time.time()
    
    p0 = multiprocessing.Process(target = worker, args=(0,))
    p1 = multiprocessing.Process(target = worker, args=(1,))
    p2 = multiprocessing.Process(target = worker, args=(2,))
    p3 = multiprocessing.Process(target = worker, args=(3,))
    p4 = multiprocessing.Process(target = worker, args=(4,))
    p5 = multiprocessing.Process(target = worker, args=(5,))
    p6 = multiprocessing.Process(target = worker, args=(6,))
    p7 = multiprocessing.Process(target = worker, args=(7,))
    p8 = multiprocessing.Process(target = worker, args=(8,))
    p9 = multiprocessing.Process(target = worker, args=(9,))
    p10 = multiprocessing.Process(target = worker, args=(10,))
    p11 = multiprocessing.Process(target = worker, args=(11,))
    p12 = multiprocessing.Process(target = worker, args=(12,))
    p13 = multiprocessing.Process(target = worker, args=(13,))
    p14 = multiprocessing.Process(target = worker, args=(14,))
    
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    p9.start()
    p10.start()
    p11.start()
    p12.start()
    p13.start()
    p14.start()

    p0.join()
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()
    p8.join()
    p9.join()
    p10.join()
    p11.join()
    p12.join()
    p13.join()
    p14.join()
    
    print('\n\n\n')
    print("#####################################")
    print("#            Data Loaded            #")
    print("#####################################")
    print(f'Elapsed Time:       {(time.time()-tik)/3600} hours')

