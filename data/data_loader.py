import time
import h5py
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, Dataset


def data_loader(foil_n, alpha):
    # Got to load in processed data into here
    alf = alpha-4
    alf_path = f'AoA_{alf}'
    vars = ["rho","rho_u","rho_v", "e", "omega"]#, "airfoil"]
    pos_vars = ["x", "y"]
    data_path = 'E:/turb_model/minimal/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
    with h5py.File(data_path, 'r') as hf:
        mesh_sz = hf[alf_path]['nodes']['rho'][()].size
        data = np.empty((len(vars), mesh_sz))
        (cl, cd) = hf[alf_path]['coeffs'][()]
        lmx, lmy = hf[alf_path]['lm']['x'][:][()], hf[alf_path]['lm']['y'][:][()]
        lm = torch.tensor(np.vstack((lmx,lmy)).T)
        xk, yk = hf[alf_path]['nodes']['x'][:][()], hf[alf_path]['nodes']['y'][:][()]
        pos = torch.tensor(np.vstack((xk,yk)).T)
        
        #nodes
        for i in range(len(vars)):
            # print(hf[alf_path]['nodes'][vars[i]][:][()])
            data[i,:] = hf[alf_path]['nodes'][vars[i]][:][()]
        foil_geom = hf[alf_path]['nodes']["airfoil"][:][()]
        g_x = torch.tensor(np.transpose(data)).to(torch.float32)
        
        #edges
        edge_arr = np.array(hf[alf_path]['edges'][()])
        edge_data = torch.tensor(np.transpose(edge_arr)).to(torch.int64)
    
    Re = 3e6
    Ma_inf = 0.1
    rho_inf = 1
    rho_u_inf = Ma_inf * math.cos(alf)
    rho_v_inf = Ma_inf * math.sin(alf)
    c = 340.15
    vel_inf = Ma_inf * c
    u_inf = vel_inf * math.cos(alf)
    v_inf = vel_inf * math.sin(alf)
    
    #["rho","rho_u","rho_v", "e", "omega"]
    d_init = data
    d_init[0,:] = rho_inf
    d_init[1,:] = rho_u_inf
    d_init[2,:] = rho_v_inf
    d_init[3,:] = 0
    print(f'{d_init=}')

    # for i in range(1,4):
    #     d_init[i,:] = 0
    ic_x = torch.tensor(np.transpose(d_init)).to(torch.float32)
    
    print(f'{g_x=}')
    print(f'{ic_x=}')
    
    args = {
        'pos': pos,
        'x': ic_x,
        'edge_index': edge_data,
        'y': g_x,
        'lm': lm,
        'Re': Re,
        'Ma': Ma_inf,
        'rho_u': rho_u_inf, 
        'rho_v': rho_v_inf, 
        'u': u_inf, 'v': v_inf, 
        'cl':cl, 'cd':cd, 
        'foil_n':foil_n, 
        'alpha':alf, 
        'foil_geom':foil_geom}
    
    # g_x = np.transpose(g_x)
    # ic_x = np.transpose(ic_x.to(torch.float32))
    graph_data = Data(**args)
    
    # print('t1')
    # if alf == -4:
    #     print('------------')
    #     print(f'{foil_n=}   {alf=}')
    #     print(f'{ic_x.size()=}')
    #     print(f'{edge_data.size()=}')
        # for i in range(5):
        #     print(f'x {i} {graph_data.x[:,i]}')
        #     print(f'y {i} {graph_data.y[:,i]}')
    # plotter = False
    # if plotter:
    #     node_colours = ['red' if data[7,node] else 'blue' for node in range(len(data[7]))]
    #     plt.figure()
    #     print('Making Graph')
    #     nx.draw(G, pos=(data[0,:], data[1,:]), node_color = node_colours, node_size=10) #THIS WON'T WORK
    #     plt.show()
    
    return graph_data


if __name__ == "__main__":
    data_path = 'E:/turb_model/Re_3M/Airfoil_0000.h5'
    
    with h5py.File(data_path, 'r') as hf:
        top_groups = [k for k in hf["AoA_0"]['nodes'].keys()]
        print(top_groups)
    
    foil_n = 0
    alpha = 0
    tik = time.time()
    data_loader(foil_n, alpha)
    print(f'Time to load 1 AoA:     {time.time()-tik}s')