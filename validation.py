from tqdm import tqdm
import numpy as np
import torch
import os
import json
import h5py

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

def print_lm(foils):
    sv_path = 'C:/Users/olive/Documents/Code/eXFoil/eX-Foil/metrics/BestWorstFoils'
    lmxs, lmys = [], []
    for foil_n in foils:
        data_path = 'E:/turb_model/Re_3M/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
        with h5py.File(data_path) as hf:
            lmxs.append(hf['AoA_0']['lm']['x'][:][()])
            lmys.append(hf['AoA_0']['lm']['y'][:][()])
    lims = (-0.25, 0.4)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 32,
        'axes.titlesize': 32,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30
        })
    fig1, ax1 = plt.subplots(2,2, sharex=False, figsize=(15,10))
    fig1.suptitle('Most Accurate Validation Foils')
    ax1[0,0].set_title('Foil '+str(foils[0]))
    ax1[0,0].plot(lmxs[0][:], lmys[0][:], ls='-')
    ax1[0,0].set_ylim(lims)
    ax1[0,1].set_title('Foil '+str(foils[1]))
    ax1[0,1].plot(lmxs[1][:], lmys[1][:], ls='-')
    ax1[0,1].set_ylim(lims)
    ax1[1,0].set_title('Foil '+str(foils[2]))
    ax1[1,0].plot(lmxs[2][:], lmys[2][:], ls='-')
    ax1[1,0].set_ylim(lims)
    ax1[1,1].set_title('Foil '+str(foils[3]))
    ax1[1,1].plot(lmxs[3][:], lmys[3][:], ls='-')
    ax1[1,1].set_ylim(lims)
    fig1.savefig(os.path.join(sv_path, 'BestFoils.png'))
    fig2, ax2 = plt.subplots(2,2, sharex=False, figsize=(15,10))
    fig2.suptitle('Least Accurate Validation Foils')
    ax2[0,0].set_title('Foil '+str(foils[4]))
    ax2[0,0].plot(lmxs[4][:], lmys[4][:], ls='-')
    ax2[0,0].set_ylim(lims)
    ax2[0,1].set_title('Foil '+str(foils[5]))
    ax2[0,1].plot(lmxs[5][:], lmys[5][:], ls='-')
    ax2[0,1].set_ylim(lims)
    ax2[1,0].set_title('Foil '+str(foils[6]))
    ax2[1,0].plot(lmxs[6][:], lmys[6][:], ls='-')
    ax2[1,0].set_ylim(lims)
    ax2[1,1].set_title('Foil '+str(foils[7]))
    ax2[1,1].plot(lmxs[7][:], lmys[7][:], ls='-')
    ax2[1,1].set_ylim(lims)
    fig2.savefig(os.path.join(sv_path, 'WorstFoils.png'))
    plt.show()


def val_run(num_foils, epochs, name_mod, loader, coef_norm, foil_range, n_val_foils=60):
    use_cuda = torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    
    model_path = os.path.join('metrics', f'{num_foils}_foils', f'{epochs}_epochs', name_mod, 'model')# name_mod)
    model = torch.load(model_path)
    
    val_outs, val_loss, _, val_surf_var, val_vol_var, val_surf, val_vol, infer_time = test(device = device,model = model, test_loader=loader)
    infer_avg = np.array(infer_time).mean()
    
    cl_rmse, cl_mse, out_list = lift_ceof(val_outs, coef_norm)
    
    sv_path = os.path.join('metrics', 'validation')
    if not os.path.exists(sv_path):
        os.mkdir(sv_path)
    with open(os.path.join(sv_path, str(num_foils)+'_foils_verif_log.json'), 'a') as f:
                json.dump(
                    {
                        'Model': name_mod,
                        'RMSE_Total': np.sqrt(val_loss),
                        'cl_rmse': cl_rmse,
                        'val_loss_total': val_loss,
                        'val_loss_surf': val_surf,
                        'rmse_surf': np.sqrt(val_surf),
                        'val_loss_surf_var': val_surf_var,
                        'val_loss_vol': val_vol,
                        'rmse_vol': np.sqrt(val_vol),
                        'val_loss_vol_var': val_vol_var,
                        'inference_time': infer_avg,
                    }, f, indent = 12, cls = NumpyEncoder
                )
    
    box_plt, foil_by_foil, fbf_tot = err_data(n_val_foils, out_list)
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 32,
        'axes.titlesize': 32,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'legend.fontsize': 30
        })
    
    g_path = os.path.join(sv_path, str(num_foils)+'_foils')
    if not os.path.exists(g_path):
        os.mkdir(g_path)
    fig_errbr, ax_errbr = plt.subplots(figsize = (20, 10))
    # ax_errbr.errorbar(x=range(-4,21), y=avg_rmse, xerr=None, yerr=(min_rmse, max_rmse))
    fig_errbr.suptitle(f'Validation CL RMSE trained on {num_foils} with {name_mod}')
    ax_errbr.set_ylim(0,5.2)
    ax_errbr.boxplot(x=box_plt, tick_labels=range(-4,21))#, patch_artist=True)
    ax_errbr.set_xlabel('Angle of Attack')
    ax_errbr.set_ylabel('RMSE')
    ax_errbr.set_title(f'Average RMSE:  {np.array(box_plt).mean()}')
    fig_errbr.savefig(os.path.join(sv_path, str(num_foils)+'_foils', name_mod+'_boxplot.png'), dpi = 150, bbox_inches = 'tight')
    
    fig_fbt, ax_fbt = plt.subplots(figsize = (20, 10))
    fig_fbt.suptitle(f'Foil to Foil RMSE Analysis  trained on {num_foils} with {name_mod}')
    ax_fbt.bar(foil_range, fbf_tot)
    ax_fbt.set_ylim(0,4)
    ax_fbt.set_xlabel('Foil Number')
    ax_fbt.set_ylabel('RMSE')
    ax_fbt.set_title(f'')
    fig_fbt.savefig(os.path.join(sv_path, str(num_foils)+'_foils', name_mod+'_fbf.png'), dpi = 150, bbox_inches = 'tight')
    # plt.show()
    plt.close()
    fbf_sv = os.path.join(g_path, 'foil_by_foil_' + name_mod)
    if not os.path.exists(fbf_sv):
        os.mkdir(fbf_sv)
    for foil in range(n_val_foils):
        foil_n = foil+1770
        data_path = 'E:/turb_model/Re_3M/Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
        with h5py.File(data_path, 'r') as hf:
            lmx, lmy = hf['AoA_0']['lm']['x'][:][()], hf['AoA_0']['lm']['y'][:][()]
            lm = torch.tensor(np.vstack((lmx,lmy)))
        
        fig_fbf, ax_fbf = plt.subplots(2, sharex=False, figsize = (10,15))
        # fig_fbf.suptitle(f'RMSE for C_L and Shape for Foil {foil_n} using {name_mod}, trained on {str(num_foils)} foils')
        fig_fbf.suptitle(f'{name_mod}   :   Foil {str(foil_n)}')
        ax_fbf[0].set_title('RMSE of Lift Coeff : Average = %.4f' % foil_by_foil[foil, :].mean())
        ax_fbf[0].plot(lmx, lmy, ls='-')
        ax_fbf[0].set_ylim(-0.25, 0.45)
        ax_fbf[0].get_xaxis().set_visible(False)
        ax_fbf[0].get_yaxis().set_visible(False)
        ax_fbf[1].plot(range(-4,21), foil_by_foil[foil, :], ls='-')
        ax_fbf[1].set_xlabel('Angle of Attack')
        ax_fbf[1].set_ylabel('RMSE')
        ax_fbf[1].get_xaxis().set_visible(True)
        fig_fbf.savefig(os.path.join(fbf_sv, 'Airfoil_' + '{:04d}'.format(foil_n)+'.png'))
        plt.close()
    # plt.show()


if __name__ == "__main__":
    models = ["MLP", "PointNet", "GraphSAGE", "GUNet"]
    foil_iter = [5, 20, 55, 150]
    epoch_n = 400
    data_set = []
    n_val_foils = 60
    start = 1770
    foil_load = range(start, start+n_val_foils)
    # foil_load = [1790,1795]
    data_path = 'E:/turb_model/Re_3M'
    
    # lm_foils = [1791, 1793, 1803, 1826, 1770, 1771, 1781, 1827]
    # print_lm(lm_foils)
    
    
    mean_in = np.array([ 5.7469070e-01, 1.1430148e-02, 1.0000000e+00, 9.8500215e-02, 1.2213878e-02, 1.0000000e-10 , 1.0000000e-10, -3.2870863e-02])
    std_in = np.array([0.39590213, 0.11666541, 0., 0.00167162, 0.01202377, 0.,  0. , 0.07188722])
    mean_out= np.array([0.997614324092865,0.05822429805994034,0.00889956671744585,1.7859368324279785,8.520916938781738])
    std_out = np.array([0.006046931724995375,0.04685685411095619,0.03721456229686737,0.01004050299525261,259.51287841796875])
    
    coef_norm = mean_in, std_in, mean_out, std_out
    
    lms = []
    for foil_n in tqdm(foil_load, desc='Loading Foils'):
        foil_name = 'Airfoil_' + '{:04d}'.format(foil_n) + '.h5'
        foil_path = os.path.join(data_path, foil_name)
        for alpha in range(25):
            data = data_loader(foil_n, alpha)
            data_set.append(data)
    
    data_set = normalise(data_set, coef_norm)
    
    loader = DataLoader(data_set, batch_size=1)
    pbar = tqdm(models, desc="Validating Models")
    for mod in pbar:
        for foil in foil_iter:
            val_run(num_foils=foil, epochs=epoch_n, name_mod=mod, loader=loader, coef_norm=coef_norm, foil_range=foil_load, n_val_foils=n_val_foils)