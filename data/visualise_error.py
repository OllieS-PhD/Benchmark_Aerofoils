import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import os
import h5py
import torch
import numpy as np
from tqdm import tqdm


def error_graphs(foil_n, alpha, num_foils, epochs, name_mod):
    
    path = 'E:/network_outs/' + str(num_foils) + '_foils/' + str(epochs) + '_epochs/'+name_mod+'/'
    vars = ["rho","rho_u","rho_v", "e", "omega"]
    data_path = path + 'airfoil_{:04d}'.format(foil_n) + '.h5'
    
    alf = alpha-4
    alf_path = f'aoa_{alf}'
    
    with h5py.File(data_path, 'r') as hf:
        # top_groups = [k for k in hf.keys()]
        # print(top_groups)
        mesh_sz = hf[alf_path]['x_nodes']['rho'][()].size
        y_data, x_data = np.empty((len(vars), mesh_sz)), np.empty((len(vars), mesh_sz))
        
        (cl, cd) = hf[alf_path]['coeffs'][()]
        xk, yk = hf[alf_path]['x_nodes']['x'][:][()], hf[alf_path]['x_nodes']['y'][:][()]
        pos = torch.tensor(np.vstack((xk,yk)).T)
        
        #nodes
        for i in range(len(vars)):
            # print(hf[alf_path]['nodes'][vars[i]][:][()])
            x_data[i,:] = hf[alf_path]['x_nodes'][vars[i]][:][()]
            y_data[i,:] = hf[alf_path]['y_nodes'][vars[i]][:][()]
        
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    
    #total rho_mag
    rho_mag_x = np.sqrt(np.square(x_data[1,:]) + np.square(x_data[2,:]))
    rho_mag_y = np.sqrt(np.square(y_data[1,:]) + np.square(y_data[2,:]))
    rel_rho_mag = np.subtract(rho_mag_y, rho_mag_x)
    
    data = np.transpose(np.subtract(y_data, x_data))
    
    G = nx.Graph()
    for i in range(len(pos)):
        G.add_node(i, x = xk[i], y=yk[i],  rho=data[i,0], rho_u=data[i,1], rho_v=data[i,2], e=data[i,3], omega=data[i,4], rho_mag = rel_rho_mag[i])
    
    node_values = data[:,1]
    # node_values = G.nodes[:]['rho_u']
    cmap=matplotlib.colormaps['RdBu']
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
    ax.set_title(f'Foil #{foil_n} at {alf} degrees AoA for {num_foils} foils over {epochs} epochs')
    nx.draw_networkx_nodes(G, pos=pos, node_color = node_colours, node_size=10)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Flat Error (Total Momentum)', rotation=270, labelpad=15)
    
    sv_path = path + 'airfoil_{:04d}'.format(foil_n)
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    
    fig.savefig(os.path.join(path, 'airfoil_{:04d}'.format(foil_n), f'{alpha}-AoA_{alf}.png'),dpi = 150, bbox_inches = 'tight')
    plt.close()

if __name__ == "__main__":
    num_foils = 10
    epochs = 10
    name_mod = 'MLP' # 'PointNet'
    
    
    for alpha in tqdm(range(3,6), desc = "Loading Results"):
        error_graphs(8, alpha, num_foils, epochs)
    
    plt.show()