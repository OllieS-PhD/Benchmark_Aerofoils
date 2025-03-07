import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import networkx as nx
from time import time


def post_process(val_outs):
    gidx = 0
    spec_out = val_outs[gidx].cpu()
    # for i in range(5):
    #     print(i, spec_out.x[:,i])
    pos = spec_out.pos
    data = spec_out.x
    # npedges = spec_out.edge_index.numpy()
    edges = spec_out.edge_index#tuple(map(tuple, npedges.tolist()))
    G = nx.Graph()
    for i in range(len(pos)):
        G.add_node(i, pos = pos[i],  rho=data[i,0], rho_u=data[i,1], rho_v=data[i,2], e=data[i,3], omega=data[i,4])
    for i in range(len(edges[0])):
        u_add = edges[0,i].item()
        v_add = edges[1,i].item()
        G.add_edge(u_add,v_add)
    node_values = data[:,1]
    # node_values = G.nodes[:]['rho_u']
    cmap=plt.cm.get_cmap('jet')
    # Create a Normalize object
    vmin = min(node_values)
    vmax=max(node_values)
    norm = Normalize(vmin, vmax)
    # Normalize the values
    node_colours = norm(node_values)
    node_colours = cmap(node_colours)
    
    edge_colours = []
    for u, v in G.edges():
        start_val = norm(node_values[u])
        end_val = norm(node_values[v])
        edge_color = cmap((start_val + end_val) / 2)
        edge_colours.append(edge_color)
    print('Drawing graph')
    tik = time()
    plt.figure()
    nx.draw(G, pos=pos, node_color = node_colours, edge_color = edge_colours, node_size=10)#, cmap=cmap)
    # nx.draw_networkx_edges(G, pos=pos, edge_color = edge_colours)#, cmap=cmap)#, node_size=10)
    print(f'Graph Time:     {time()-tik} seconds')
    plt.show()