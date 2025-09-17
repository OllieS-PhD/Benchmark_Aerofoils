import random
import os.path as osp
import torch
from torch_geometric.data import Data

from normalise import load_normalisation_coefs, norm_data

def data_loader(foil_n, alpha, file_path='./data/turb_model/Re_3M', trans_aug=False, graph_bn=False, norm_bn = True):
    sv_path = osp.join(file_path, f'Airfoil_{foil_n:04d}', f'AoA_{alpha}')
    if osp.exists(sv_path):
        pos = torch.load(osp.join(sv_path, 'pos.pt'), weights_only=True)
        x = torch.load(osp.join(sv_path, 'x.pt'), weights_only=True)
        y = torch.load(osp.join(sv_path, 'y.pt'), weights_only=True)
        surf = torch.load(osp.join(sv_path, 'surf.pt'), weights_only=True)
        lm = torch.load(osp.join(sv_path, 'lm.pt'), weights_only=True)
        coeffs = torch.load(osp.join(sv_path, 'coeffs.pt'), weights_only=True)

        if graph_bn:
            edge_index = torch.load(osp.join(sv_path, 'edge_index.pt'))
        
        if trans_aug:
            aug_range = 200
            # Randomly translate the data
            pos[0,:] += random.uniform(-aug_range, aug_range)
            pos[1,:] += random.uniform(-aug_range, aug_range)
        
        coef_norm = load_normalisation_coefs()
        
        if graph_bn:
            data = Data(pos=pos, x=x, y=y, surf=surf, lm=lm, edge_index=edge_index, cl=coeffs[0], cd=coeffs[1], foil_n=foil_n, alpha=(alpha-4))
        else:
            data = Data(pos=pos, x=x, y=y, surf=surf, lm=lm, cl=coeffs[0], cd=coeffs[1], foil_n=foil_n, alpha=(alpha-4))
        
        if norm_bn:
            return norm_data(data, coef_norm)
        else:
            return data
    else:
        print('\nERROR:')
        print(f'No data found for Airfoil {foil_n:04d} at AoA {alpha}.\n')
        return None

