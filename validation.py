from tqdm import tqdm
import numpy as np
import torch
import os
import json

from normalise import normalise
from post_proc.postproc_loader import data_loader
from post_proc.panel_method import lift_ceof, err_data
from train import test

import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def val_run(num_foils, epochs, name_mod, loader, coef_norm):
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    
    model_path = os.path.join('metrics', f'{num_foils}_foils', f'{epochs}_epochs', name_mod, 'model')# name_mod)
    model = torch.load(model_path)
    
    val_outs, val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol = test(device = device,model = model, test_loader=loader)
    
    cl_rmse, cl_mse, out_list = lift_ceof(val_outs, coef_norm)
    
    sv_path = os.path.join('metrics', 'validation')
    with open(os.path.join(sv_path, str(num_foils)+'_foils_verif_log.json'), 'a') as f:
                json.dump(
                    {
                        'Model': name_mod,
                        'RMSE_Total': np.sqrt(val_loss),
                        'cl_rmse': cl_rmse,
                        'val_loss_total': val_loss,
                        'val_loss_surf': val_surf,
                        'val_loss_surf_var': val_surf_var,
                        'val_loss_vol': val_vol,
                        'val_loss_vol_var': val_vol_var,
                    }, f, indent = 12, cls = NumpyEncoder
                )
    
    avg_rmse, max_rmse, min_rmse, box_plt = err_data(out_list)
    fig_errbr, ax_errbr = plt.subplots(figsize = (20, 5))
    # ax_errbr.errorbar(x=range(-4,21), y=avg_rmse, xerr=None, yerr=(min_rmse, max_rmse))
    ax_errbr.boxplot(x=box_plt, tick_labels=range(-4,21))#, patch_artist=True)
    ax_errbr.set_xlabel('Angle of Attack')
    ax_errbr.set_ylabel('RMSE')
    ax_errbr.set_title(f'Validation CL RMSE for {str(num_foils)} foils with {name_mod}')
    fig_errbr.savefig(os.path.join(sv_path, str(num_foils)+'_foils', name_mod+'.png'), dpi = 150, bbox_inches = 'tight')
    # plt.show()


if __name__ == "__main__":
    models = ["MLP", "PointNet", "GraphSAGE", "GUNet"]
    foil_iter = [5, 20, 55, 150]
    epoch_n = 400
    data_set = []
    foil_load = range(1770, 1830)
    # foil_load = [1790,1795]
    data_path = 'E:/turb_model/Re_3M_ValSet'
    
    mean_in = np.array([ 5.7469070e-01, 1.1430148e-02, 1.0000000e+00, 9.8500215e-02, 1.2213878e-02, 1.0000000e-10 , 1.0000000e-10, -3.2870863e-02])
    std_in = np.array([0.39590213, 0.11666541, 0., 0.00167162, 0.01202377, 0.,  0. , 0.07188722])
    mean_out= np.array([0.997614324092865,0.05822429805994034,0.00889956671744585,1.7859368324279785,8.520916938781738])
    std_out = np.array([0.006046931724995375,0.04685685411095619,0.03721456229686737,0.01004050299525261,259.51287841796875])
    
    coef_norm = mean_in, std_in, mean_out, std_out
    
    
    for foil_n in tqdm(foil_load, desc='Loading Foils'):
        foil_name = 'Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
        foil_path = os.path.join(data_path, foil_name)
        for alpha in range(25):
            data_set.append(data_loader(foil_n, alpha))
    
    data_set = normalise(data_set, coef_norm)
    
    loader = DataLoader(data_set, batch_size=1)
    
    for mod in tqdm(models, desc="Validating Models"):
        for foil in foil_iter:
            val_run(num_foils=foil, epochs=epoch_n, name_mod=mod, loader=loader, coef_norm=coef_norm)