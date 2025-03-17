import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from tqdm import tqdm
import networkx as nx
from time import time
import numpy
import h5py
import os


def post_process(val_outs, name_mod, hparams, num_foils=30):
    vars = ["x","y","rho","rho_u","rho_v", "e", "omega"]
    data_path = 'E:/network_outs/' + str(num_foils) + '_foils/' + str(hparams['nb_epochs']) + '_epochs/'+name_mod+'/'
    
    for gidx in tqdm(val_outs, desc='Saving Validation Outputs'):
        spec_out = gidx.cpu()
        # for i in range(5):
        #     print(i, spec_out.x[:,i])
        pos = spec_out.pos
        data = spec_out.x
        data_y = spec_out.y
        alpha = int(spec_out.alpha)
        foil_n = int(spec_out.foil_n)
        # npedges = spec_out.edge_index.numpy()
        # edges = spec_out.edge_index#tuple(map(tuple, npedges.tolist()))
        xk, yk = pos[:,0], pos[:,1]
        G = nx.Graph()
        Y = nx.Graph()
        for i in range(len(pos)):
            G.add_node(i, x = xk[i], y=yk[i],  rho=data[i,0], rho_u=data[i,1], rho_v=data[i,2], e=data[i,3], omega=data[i,4])
            Y.add_node(i, x = xk[i], y=yk[i],  rho=data_y[i,0], rho_u=data_y[i,1], rho_v=data_y[i,2], e=data_y[i,3], omega=data_y[i,4])
        # for i in range(len(edges[0])):
        #     u_add = edges[0,i].item()
        #     v_add = edges[1,i].item()
        #     G.add_edge(u_add,v_add)
        
        file_path = data_path +'airfoil_'+str('{:04d}'.format(foil_n))+'.h5'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            print(f'\nFile path {data_path} created...')
        grp_path = 'aoa_'+str(alpha)
        with h5py.File(file_path, 'a') as hf:
            grp = hf.create_group(grp_path)
            grp.create_dataset('coeffs', data = (spec_out.cl,spec_out.cd))
            # grp.create_dataset('edges', data=spec_out.edge_index)#, compression = "gzip")
            
            # Loop through each attribute
            node_group = grp.create_group('x_nodes')
            for att in vars:        #tqdm(att_vars, desc='Creating Datasets'):
                att_data = [G.nodes[nid][att] for nid in G.nodes()]
                node_group.create_dataset(att, data=att_data)
            
            y_group = grp.create_group('y_nodes')
            for att in vars:        #tqdm(att_vars, desc='Creating Datasets'):
                att_data = [Y.nodes[nid][att] for nid in Y.nodes()]
                y_group.create_dataset(att, data=att_data)
    
    print("Validation Data Saved")
    # node_values = data[:,1]
    # # node_values = G.nodes[:]['rho_u']
    # cmap=plt.cm.get_cmap('jet')
    # # Create a Normalize object
    # vmin = min(node_values)
    # vmax=max(node_values)
    # norm = Normalize(vmin, vmax)
    # # Normalize the values
    # node_colours = norm(node_values)
    # node_colours = cmap(node_colours)
    
    # edge_colours = []
    # for u, v in G.edges():
    #     start_val = norm(node_values[u])
    #     end_val = norm(node_values[v])
    #     edge_color = cmap((start_val + end_val) / 2)
    #     edge_colours.append(edge_color)
    # print('Drawing graph')
    # tik = time()
    # plt.figure()
    # nx.draw(G, pos=pos, node_color = node_colours, edge_color = edge_colours, node_size=10)#, cmap=cmap)
    # # nx.draw_networkx_edges(G, pos=pos, edge_color = edge_colours)#, cmap=cmap)#, node_size=10)
    # print(f'Graph Time:     {time()-tik} seconds')
    # plt.show()