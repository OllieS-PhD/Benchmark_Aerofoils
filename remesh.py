from scipy.spatial import Delaunay
from tqdm import tqdm

import torch
import numpy as np

def remesh(pos, lm):
    pos = pos.cpu()
    lm = lm.cpu()
    tri = Delaunay(pos)
    edges = set()
    
    for simplex in tri.simplices: # tqdm(tri.simplices, desc="Adding Edges"):
        if not frozenset((simplex[0], simplex[1])) in edges: edges.add((simplex[0], simplex[1]))
        if not frozenset((simplex[1], simplex[2])) in edges: edges.add((simplex[1], simplex[2]))
        if not frozenset((simplex[2], simplex[0])) in edges: edges.add((simplex[2], simplex[0]))
    edges = list(map(list, edges))
    edges = torch.tensor(np.transpose(edges)).cuda()
    edges = edges.type(torch.int64)
    # print('-----------------------')
    # print(f'Downsampled {len(edges)=}   {len(edges[0])=}')
    # print(f'{edges.dtype=}')
    # print('-----------------------')
    return edges