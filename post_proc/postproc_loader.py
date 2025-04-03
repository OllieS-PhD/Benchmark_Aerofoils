import time
import h5py
import numpy as np
import math
import torch
import torch_geometric.nn as nng
from torch_geometric.data import Data


def data_loader(foil_n, alpha):
    # Got to load in processed data into here
    alf = alpha-4
    alf_path = f'AoA_{alf}'
    vars = ["rho","rho_u","rho_v", "e", "omega", "dist"]#, "airfoil"]
    pos_vars = ["x", "y"]
    data_path = 'E:/turb_model/Re_3M_ValSet/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
    with h5py.File(data_path, 'r') as hf:
        mesh_sz = hf[alf_path]['nodes']['rho'][()].size
        data = np.empty((len(vars), mesh_sz))
        (cl, cd) = hf[alf_path]['coeffs'][()]
        lmx, lmy = hf[alf_path]['lm']['x'][:][()], hf[alf_path]['lm']['y'][:][()]
        lm = torch.tensor(np.vstack((lmx,lmy)).T)
        xk, yk = hf[alf_path]['nodes']['x'][:][()], hf[alf_path]['nodes']['y'][:][()]
        # print('------------------------------------')
        # print(f'Foil    {foil_n}        Alpha   {alf}')
        # print(f'{len(data)}     {len(data[0])}')
        #nodes
        for i in range(len(vars)):
            # print(hf[alf_path]['nodes'][vars[i]][:][()])
            data[i,:] = hf[alf_path]['nodes'][vars[i]][:][()]
        foil_geom = hf[alf_path]['nodes']["airfoil"][:][()]
        # np.set_printoptions(threshold=sys.maxsize)
        # print([xk[i] for i in range(len(xk)) if xk[i] > 100])
        # print([data[5,i] for i in range(len(data[0])) if data[5,i] > 10])
        # quit()
    foil_geom = np.array(foil_geom)
    data = np.array(data)
    data[5,:] = -data[5,:]
    
    radius = 0.7 # 2 # 4 # 7
    
    del_list = [i for i in range(mesh_sz) if np.sqrt( np.square(xk[i]-0.5) + np.square(yk[i]) ) > radius]
    data = np.delete(data, del_list, axis=1)
    foil_geom = torch.tensor(np.delete(foil_geom.T, del_list))
    pos = np.delete(np.vstack((xk,yk)), del_list, axis=1)
    for i in range(len(foil_geom)):
        if pos[0,i] < 0:
            foil_geom[i] == False
    # print(f'{len(data)}     {len(data[0])}')
        
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
    
    eps = 1e-10
    
    d_init = np.copy(data)
    d_init[0,:] = rho_inf
    d_init[1,:] = rho_u_inf
    d_init[2,:] = rho_v_inf
    d_init[3,:] = eps
    d_init[4,:] = eps
    
    d_init = np.vstack((pos, d_init))
    pos = torch.tensor(pos.T)
    data = np.delete(data, 5, 0)
    
    edge_data = nng.radius_graph(x = pos, r = 0.05, loop = True, max_num_neighbors = int(5)).cpu()
    
    g_x = torch.tensor(np.transpose(data)).to(torch.float32)
    ic_x = torch.tensor(np.transpose(d_init)).to(torch.float32)
    
    args = {
        'pos': pos,
        'x': ic_x,
        'edge_index': edge_data,
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
    graph_data = Data(**args)
    
    
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