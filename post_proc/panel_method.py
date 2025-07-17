import math
import numpy as np

import torch
import torch.nn as nn

from tqdm import tqdm

from normalise import denormalise_ys
from scipy.integrate import quad
from scipy.spatial import Delaunay

def integrate(vertices, om_values):
    # Triangulate polygon
    tri = Delaunay(vertices)
    triangles = vertices[tri.simplices]
    om_triangles = om_values[tri.simplices]
    
    total_integral = 0.0
    
    for triangle, om in zip(triangles, om_triangles):
        # Compute area using determinant
        v1, v2, v3 = triangle
        area = 0.5 * abs(
            (v2[0] - v1[0]) * (v3[1] - v1[1]) -
            (v3[0] - v1[0]) * (v2[1] - v1[1])
        )
        # Average omega and accumulate integral
        avg_om = np.mean(om)
        total_integral += area * avg_om
    
    return total_integral



def lift_ceof(val_outs, coef_norm):
    # Calculate Lift Coefficient from data
    outs = denormalise_ys(val_outs, coef_norm)
    iter = 0
    out_list = []
    mse = 0
    rmse = 0
    cl_list = []
    cl_gt_list = []
    for d in outs:
        data = d[0]
        cl_target = data.cl
        preds = data.x
        rho = preds[:,0]
        rho_u = preds[:,1]
        rho_y = preds[:,2]
        e = preds[:,3]
        omega = preds[:,4]
        
        x = data.pos[:,0]
        
        a = 340.15
        U = 0.1 * a
        surf = data.surf
        
        x_foil = np.array(x[surf])
        del_list = [i for i in range(len(x_foil)) if x_foil[i]<0]
        
        pos = np.array(data.pos[surf])
        omega_foil = np.array(omega[surf])
        
        pos = np.delete(pos, del_list, axis=0)
        omega_foil = np.delete(omega_foil, del_list, axis=0)
        
        # panel_lengths = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        # # print(panel_lengths)
        # Gam = np.sum(omega_foil * panel_lengths)
        
        Gam = integrate(pos, omega_foil)
        # print(f'\n{Gam=}')
        
        # Gam = np.trapz(omega_foil, x_foil)
        cl = torch.tensor([(2*Gam) / U])
        # print(f'\n{cl=}')
        # print(f'\n{cl_target=}')
        # print(f'{torch.Size(cl)=},  {torch.Size(cl_target)=}')
        # print('--------------')
        # print(cl, cl_target)
        
        loss = nn.MSELoss()
        mse_loss = loss(cl, cl_target)
        root = math.sqrt(mse_loss)
        
        mse += mse_loss.cpu().numpy()
        rmse += root
        
        # print(f'cl_Target:  {cl_target}      cl_pred:    {cl}')
        
        out_list.append([data.foil_n, data.alpha, root, cl_target, cl])
        iter += 1
    
    return rmse/iter, mse/iter, out_list

def err_data(n_val_foils, out_list):
    box_plt = []
    fbf = []
    fbf_tot = []
    for alpha in range(25):
        alf = alpha-4
        err_list = []
        for i in range(len(out_list)):
            if out_list[i][1] == alf:
                err_list.append(out_list[i][2])
        box_plt.append(err_list)
        
        # foil-by-foil analysis
    fbf= np.zeros((n_val_foils, 25))
    for foil in range(n_val_foils):
        foil_n = foil + 1770
        for i in range(len(out_list)):
            for alpha in range(25):
                alf = alpha-4
                if out_list[i][0] == foil_n and out_list[i][1] == alf:
                    fbf[foil, alpha] = out_list[i][2]    
    
    for foil_n in range(n_val_foils):
        foil = foil_n + 1770
        current = []
        for i in range(len(out_list)):
            if foil == out_list[i][0]:
                current.append(out_list[i][2])
        
        fbf_tot.append(np.array(current).mean())
    
    return box_plt, fbf, fbf_tot