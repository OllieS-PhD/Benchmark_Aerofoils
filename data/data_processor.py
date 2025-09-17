# Imports
import os
import os.path as osp
import sys
import h5py
import torch
import torch_geometric.nn as nng
import numpy as np
import pandas as pd
import altair as alt
import time
import math
from tqdm import tqdm

import multiprocessing

from shapely import distance
from shapely.geometry import Polygon, Point


# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=sys.maxsize)
alt.data_transformers.disable_max_rows()

def dataSorter(foil_n, alpha):
    """
    data_path_load is hardcoded, you will need to change this to the path where you saved the raw data.
    """
    data_path_load = 'O:/WindAI_Data/raw/airfoil_2k_data.h5'
    model='turb_model'
    Re='Re06000000' # 'Re03000000' 'Re09000000'
    var = ["x","y","rho","rho_u","rho_v", "e", "omega"]
    var_sz = len(var)
    #print(range(len(dtype)))
    # tic = time.time()
    with h5py.File(data_path_load, 'r') as hf:
        lm = hf['shape']['landmarks'][foil_n][()]
        mesh_sz = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)]['x'][:,alpha][()].size
        data = np.empty((var_sz, mesh_sz))
        x = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)]['x'][:,alpha][()]
        y = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)]['y'][:,alpha][()]
        for i in range(var_sz-2):
            data[i,:] = hf[model][Re]['flow_field']['{:04d}'.format(foil_n)][var[i+2]][:,alpha][()]
        #data = torch.Tensor(np.swapaxes(data, 0, 1))
        
        alf = hf[model][Re]['alpha'][alpha][()]
        cl = hf[model][Re]['C_l'][foil_n,alpha][()]
        cd = hf[model][Re]['C_d'][foil_n,alpha][()]
    
    airfoil = []
    dist = np.zeros(mesh_sz)
    # for i in range(len(lm)):
    #     G.add_node(mesh_sz+i, rho = )
    pos = np.vstack((x, y))

    aero_poly = Polygon(lm)
    
    buff_dist = 3.5e-6
    buff_poly = aero_poly.buffer(buff_dist)
    n_foil_points = 0
    for nid in range(mesh_sz):
        point = Point( (x[nid], y[nid]) )
        dist[nid] = distance(aero_poly, point)
        if buff_poly.contains( point ):
            airfoil.append(True)
            n_foil_points +=1
        else:
            airfoil.append(False)
    data[4,:] = airfoil
    data[5,:] = -dist
    radius = 0.7 
    del_list = [i for i in range(mesh_sz) if np.sqrt( np.square(x[i]-0.5) + np.square(y[i]) ) > radius]

    data = np.delete(data, del_list, axis=1)
    pos = np.delete(pos, del_list, axis=1)
    
    
    Ma_inf = 0.1
    rho_inf = 1
    alf_rad = math.radians(alf)
    sig_u_inf = Ma_inf * math.cos(alf_rad)
    sig_v_inf = Ma_inf * math.sin(alf_rad)
    
    d_init = np.copy(data)
    d_init[0,:] = rho_inf
    d_init[1,:] = sig_u_inf
    d_init[2,:] = sig_v_inf
    d_init = np.delete(d_init, 3, 0)  # Remove the 'e' variable
    d_init = np.delete(d_init, 4, 0)  # Remove the 'omega' variable
    
    
    d_init = np.vstack((pos, d_init))
    pos = torch.tensor(pos.T)
    data = np.delete(data, 5, 0)
    
    edge_data = nng.radius_graph(x = pos, r = 0.05, loop = True, max_num_neighbors = int(5)).cpu()
    g_x = torch.tensor(np.transpose(data)).to(torch.float32)
    ic_x = torch.tensor(np.transpose(d_init)).to(torch.float32)
    
    edge_data = nng.radius_graph(x = pos, r = 0.05, loop = True, max_num_neighbors = int(5)).cpu()
    # print(att_vars)
    if Re == 'Re03000000':
        data_path_save = './data/turb_model/Re_3M'
    elif Re == 'Re06000000':
        data_path_save = './data/turb_model/Re_6M'
    elif Re == 'Re09000000':
        data_path_save = './data/turb_model/Re_9M'
    else:
        print('Error: Re not recognised')
        return
    
    sv_path = os.path.join(data_path_save, f'Airfoil_{foil_n:04d}', f'AoA_{alpha}')
    if not osp.exists(sv_path):
        os.makedirs(sv_path)
    # print(f'Saving data to {sv_path}')
    torch.save(pos, osp.join(sv_path, 'pos.pt'))
    torch.save(ic_x, osp.join(sv_path, 'x.pt'))
    torch.save(g_x, osp.join(sv_path, 'y.pt'))
    torch.save(airfoil, osp.join(sv_path, 'surf.pt'))
    torch.save(lm, osp.join(sv_path, 'lm.pt'))
    torch.save(edge_data, osp.join(sv_path, 'edge_index.pt'))
    torch.save(torch.tensor([cl, cd]), osp.join(sv_path, 'coeffs.pt'))
    
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

