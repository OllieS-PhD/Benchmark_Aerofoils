import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import root_mean_squared_error as rmse

import os
import h5py
import torch
import numpy as np
from tqdm import tqdm


def error_graphs(foil_n, alpha, num_foils, epochs, name_mod, var, folder='norm', type='err'):
    
    path = 'E:/network_outs/' + str(num_foils) + '_foils/' + str(epochs) + '_epochs/'+name_mod+'/'+'airfoil_{:04d}'.format(foil_n)+'/'
    vars = ["rho","rho_u","rho_v", "e", "omega"]
    data_path = path + folder + '.h5'
    
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
    
    RMSE = rmse(y_true=y_data, y_pred=x_data)
    
    #total rho_mag
    rho_mag_x = np.sqrt(np.square(x_data[1,:]) + np.square(x_data[2,:]))
    rho_mag_y = np.sqrt(np.square(y_data[1,:]) + np.square(y_data[2,:]))
    rel_rho_mag = np.absolute(np.subtract(rho_mag_y, rho_mag_x))
    
    
    #Error Calc
    if type == 'err':
        data = np.transpose( np.absolute(np.subtract(y_data, x_data)/x_data) )
        t_out = 'error'
    elif type =='y':
        data=np.transpose(y_data)
        rel_rho_mag = rho_mag_y
        t_out = 'ground_truth'
    elif type =='x':
        data=np.transpose(x_data)
        rel_rho_mag = rho_mag_x
        t_out = 'prediction'
    else:
        print('Invalid Data Type...')
        return
    # if var == 'rho_u':
    #     for i in range(5):
    #         print(f'\n{data[:,i]=}')
    G = nx.Graph()
    for i in range(len(data)):
        G.add_node(i, x = xk[i], y=yk[i],  rho=data[i,0], rho_u=data[i,1], rho_v=data[i,2], e=data[i,3], omega=data[i,4], rho_mag = rel_rho_mag[i])
    
    # limit = 100
    # r_nodes = [node for node, data in G.nodes(data=True) if data["rho_u"] > limit or data["rho_v"] > limit]
    # G.remove_nodes_from(r_nodes)
    
    # node_values = data[:,1]
    node_values = [G.nodes[nid][var] for nid in G.nodes()]
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
    ax.set_title(f'Foil #{foil_n} at {alf} degrees AoA for {num_foils} foils over {epochs} epochs:     {name_mod}')
    fig.text(x=0,y=1,s=f'RMSE:    {RMSE}')
    nx.draw_networkx_nodes(G, pos=pos, node_color = node_colours, node_size=5)
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(f'Relative Momentum ({var})', rotation=270, labelpad=15)
    
    sv_path = path + f'graphs_{folder}/' + var +'/'+ t_out
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    
    fig.savefig(os.path.join(path, f'graphs_{folder}' ,var, t_out, f'{alpha}-AoA_{alf}.png'),dpi = 150, bbox_inches = 'tight')
    # if var=='rho_u':
    #     plt.show()
    plt.close()
    return RMSE

if __name__ == "__main__":
    num_foils = 20
    num_epochs = 400
    name_mod = ['MLP', 'PointNet', 'GraphSAGE']#, 'GUNet']
    proc_vars = ['rho_u', 'rho_v', 'rho_mag']
    types = ['y', 'x', 'err']
    val_set = [14, 15, 16, 17, 18, 19]
    for model in name_mod:
        for var in proc_vars:
            for foil_n in val_set:
                for alpha in tqdm(range(24), desc=f'{model} {var} foil_num{foil_n}'):
                    for type in types:
                        error_graphs(foil_n, alpha, num_foils, num_epochs, model, var, folder='norm', type=type)
                        error_graphs(foil_n, alpha, num_foils, num_epochs, model, var, folder='de_norm', type=type)
    # error_graphs(18, 4, 20, 50, 'PointNet', 'rho_u')
    # # plt.show()
    # model = 'MLP'
    # foil_n = 4
    # alpha = 10
    # for var in proc_vars:
    #     # for alpha in tqdm(range(24), desc=f'{model} {var} foil_num{foil_n}'):
    #     error_graphs(foil_n, alpha, num_foils, num_epochs, model, var, folder ='de_norm')
    #     error_graphs(foil_n, alpha, num_foils, num_epochs, model, var, folder ='norm')
    
    
    # plt.show()