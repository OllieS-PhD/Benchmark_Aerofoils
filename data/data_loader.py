import time
import h5py
import numpy as np
import pandas as pd
import sys
import math
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data, Dataset
from sklearn.preprocessing import MinMaxScaler


def data_loader(foil_n, alpha):
    # Got to load in processed data into here
    alf = alpha-4
    alf_path = f'AoA_{alf}'
    vars = ["rho","rho_u","rho_v", "e", "omega", "dist"]#, "airfoil"]
    pos_vars = ["x", "y"]
    data_path = 'E:/turb_model/Re_3M/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
    with h5py.File(data_path, 'r') as hf:
        mesh_sz = hf[alf_path]['nodes']['rho'][()].size
        data = np.empty((len(vars), mesh_sz))
        (cl, cd) = hf[alf_path]['coeffs'][()]
        lmx, lmy = hf[alf_path]['lm']['x'][:][()], hf[alf_path]['lm']['y'][:][()]
        lm = torch.tensor(np.vstack((lmx,lmy)).T)
        xk, yk = hf[alf_path]['nodes']['x'][:][()], hf[alf_path]['nodes']['y'][:][()]
        
        for i in range(len(vars)):
            data[i,:] = hf[alf_path]['nodes'][vars[i]][:][()]
        foil_geom = hf[alf_path]['nodes']["airfoil"][:][()]
        
    foil_geom = np.array(foil_geom)
    data = np.array(data)
    data[5,:] = -data[5,:]
    
    radius = 300 #0.7 
    
    del_list = [i for i in range(mesh_sz) if np.sqrt( np.square(xk[i]-0.5) + np.square(yk[i]) ) > radius]

    del_list = [i for i in range(mesh_sz) if np.sqrt( np.square(xk[i]) + np.square(yk[i]) ) > radius]
    data = np.delete(data, del_list, axis=1)
    foil_geom = torch.tensor(np.delete(foil_geom.T, del_list))
    pos = np.delete(np.vstack((xk,yk)), del_list, axis=1)
    # print(f'{len(data)}     {len(data[0])}')
    pos = pos.T
    
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    dist = data[5,:] 
    G = nx.Graph()
    for i in range(len(dist)):
        G.add_node(i, pos=pos[i], dist=dist[i])
    node_values = [G.nodes[nid]['dist'] for nid in G.nodes()]
    cmap=matplotlib.colormaps['RdBu_r']
    # Create a Normalize object
    vmin = min(node_values)
    vmax = max(node_values)
    norm = Normalize(vmin, vmax)
    
    sm = ScalarMappable(cmap=cmap, norm = norm)
    sm.set_array([])
    
    # Normalize the values
    node_colours = norm(node_values)
    node_colours = cmap(node_colours)
    fig, ax = plt.subplots(figsize = (20, 10))
    ax.set_title(f'Radius = {radius} chord lengths')
    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colours, node_size=5)
    cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label(f'Dist ({dist})', rotation=270, labelpad=15)
    fig.savefig(f'./data/r={radius}_dist.png',dpi = 150, bbox_inches = 'tight')
    quit()
    plt.show()
        #edges
        # edge_arr = np.array(hf[alf_path]['edges'][()])
        # edge_data = torch.tensor(np.transpose(edge_arr)).to(torch.int64)
    # print([foil_geom[i] for i in range(len(foil_geom)) if foil_geom[i] == True])
    # for i in range(len(data[0])):
    #     if data[1,i]==0 and data[2,i]==0:
    #         print(data[1,i], data[2,i]) 
    
    Re = 3e6
    Ma_inf = 0.1
    rho_inf = 1
    alf_rad = math.radians(alf)
    rho_u_inf = Ma_inf * math.cos(alf_rad)
    rho_v_inf = Ma_inf * math.sin(alf_rad)
    c = 340.15
    vel_inf = Ma_inf * c
    u_inf = vel_inf * math.cos(alf_rad)
    v_inf = vel_inf * math.sin(alf_rad)
    
    #["rho","rho_u","rho_v", "e", "omega"]
    eps = 1e-10
    # print(f'{data=}')
    d_init = np.copy(data)
    d_init[0,:] = rho_inf
    d_init[1,:] = rho_u_inf
    d_init[2,:] = rho_v_inf
    d_init[3,:] = eps
    d_init[4,:] = eps
    
    d_init = np.vstack((pos, d_init))
    pos = torch.tensor(pos.T)
    data = np.delete(data, 5, 0)

    g_x = torch.tensor(np.transpose(data)).to(torch.float32)
    ic_x = torch.tensor(np.transpose(d_init)).to(torch.float32)
    

    
    args = {
        'pos': pos,
        'x': ic_x,
        # 'edge_index': edge_data,
        'y': g_x, 
        'surf':foil_geom,
        
        
        
        'lm': lm,
        'Re': Re,
        'Ma': Ma_inf,
        'rho_u': rho_u_inf, 
        'rho_v': rho_v_inf, 
        'u': u_inf, 'v': v_inf, 
        'cl':cl, 'cd':cd, 
        'foil_n':foil_n, 
        'alpha':alf}
    
    # g_x = np.transpose(g_x)
    # ic_x = np.transpose(ic_x.to(torch.float32))
    graph_data = Data(**args)
    
    # print(graph_data)
    # quit()
    
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
    
    # with h5py.File(data_path, 'r') as hf:
    #     top_groups = [k for k in hf["AoA_0"]['nodes'].keys()]
    #     print(top_groups)
    
    foil_n = 0
    alpha = 7
    tik = time.time()
    g = data_loader(foil_n, alpha)
    print(f'{g.foil_n=},     {g.alpha=}')
    print(f'{g.x=}')
    print(f'{g.y=}')
    print(f'Time to load 1 AoA:     {time.time()-tik}s')