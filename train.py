import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import time, json

import torch
import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.post_process import post_process
# import metrics

from tqdm import tqdm

from pathlib import Path
import os.path as osp

def get_nb_trainable_params(model):
    '''
    Return the number of trainable parameters
    '''
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def train(device, model, train_loader, optimizer, scheduler, criterion = 'RMSE',  reg = 1, mat_sz=5):
    model.train()
    avg_loss_per_var = torch.zeros(mat_sz, device = device)
    avg_loss = 0
    iter = 0
    
    for data in train_loader:
        data.to(device)          
        optimizer.zero_grad()
        # print(f'train_data:     {data.edge_index.size()=}')
        out = model(data)
        targets = data.y
        # print(f'{data.foil_n=}      {data.alpha=}')
        # print(f'{out}')
        if criterion == 'MSE' or criterion == 'MSE_weighted' or criterion == 'RMSE':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        
        if criterion == 'RMSE':
            loss_per_var = torch.sqrt(loss_criterion(out, targets)).mean(dim = 0)
        else:
            loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        # print(loss_per_var)
        total_loss = loss_per_var.mean()
        total_loss.backward()
        # data.x = scaler.inverse_transform(data.x)
        optimizer.step()
        scheduler.step()
        avg_loss_per_var += loss_per_var
        avg_loss += total_loss

        iter += 1
    avg_loss_var_iter = avg_loss_per_var.cpu().data.numpy()/iter
    avg_loss_iter = avg_loss.cpu().data.numpy()/iter
    # print()
    # print(f'{avg_loss_var_iter=}')
    # print(f'{out[:,4]=}')
    # print(f'{targets[:,4]=}')
    return avg_loss_iter,  avg_loss_var_iter

@torch.no_grad()
def test(device, model, test_loader, final_epoch, criterion = 'RMSE', mat_sz=5):
    model.eval()
    final_outs = []
    avg_loss_per_var = np.zeros(mat_sz)
    avg_loss = 0
    iter = 0
    for data in test_loader:      
        data.to(device)
        
        out = model(data)       
        targets = data.y                
        
        if criterion == 'MSE' or 'MSE_weighted' or 'RMSE':
            loss_criterion = nn.MSELoss(reduction = 'none')
        elif criterion == 'MAE':
            loss_criterion = nn.L1Loss(reduction = 'none')
        
        if criterion == 'RMSE':
            loss_per_var = torch.sqrt(loss_criterion(out, targets)).mean(dim = 0)
        else:
            loss_per_var = loss_criterion(out, targets).mean(dim = 0)
        loss = loss_per_var.mean()
        
        
        avg_loss_per_var += loss_per_var.cpu().numpy()
        avg_loss += loss.cpu().numpy()
        if final_epoch==True:
            data_outs = data
            data_outs.x = out
            final_outs.append(data_outs)
        iter += 1
    return final_outs, avg_loss/iter, avg_loss_per_var/iter 

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(device, train_dataset, val_dataset, Net, hparams, path, criterion = 'RMSE', reg = 1, val_iter = 10, name_mod = 'GraphSAGE', val_sample = True):
    '''
        Args:
        device (str): device on which you want to do the computation.
        train_dataset (list): list of the data in the training set.
        val_dataset (list): list of the data in the validation set.
        Net (class): network to train.
        hparams (dict): hyper parameters of the network.
        path (str): where to save the trained model and the figures.
        criterion (str, optional): chose between 'MSE', 'MAE', and 'MSE_weigthed'. The latter is the volumetric MSE plus the surface MSE computed independently. Default: 'MSE'.
        reg (float, optional): weigth for the surface loss when criterion is 'MSE_weighted'. Default: 1.
        val_iter (int, optional): number of epochs between each validation step. Default: 10.
        name_mod (str, optional): type of model. Default: 'GraphSAGE'.
    '''
    Path(path).mkdir(parents = True, exist_ok = True)
    model = Net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = hparams['lr'])
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = hparams['lr'],
            total_steps = (len(train_dataset) // hparams['batch_size'] + 1) * hparams['nb_epochs'],
        )
    val_loader = DataLoader(val_dataset, batch_size = 1)
    start = time.time()

    train_loss_list=[]
    val_loss_list=[]
    val_epochs = []

    pbar_train = tqdm(range(hparams['nb_epochs']), position=0)
    for epoch in pbar_train:    
        final_epoch = True if (epoch+1) == hparams['nb_epochs'] else False
        train_dataset_sampled = []
        for data in train_dataset:
            data_sampled = data.clone()
            # idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
            # idx = torch.tensor(idx)
            # data_sampled.pos = data_sampled.pos[idx].to(device)
            # data_sampled.x = data_sampled.x[idx].to(device)
            # data_sampled.y = data_sampled.y[idx].to(device)
            
            # data_sampled.surf = data_sampled.surf[idx]

            if name_mod != 'PointNet' and name_mod != 'MLP':
                data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
            
            train_dataset_sampled.append(data_sampled)
        train_loader = DataLoader(train_dataset_sampled, batch_size = hparams['batch_size'], shuffle = True)
        del(train_dataset_sampled)

        train_loss, _ = train(device, model, train_loader, optimizer, lr_scheduler, criterion, reg = reg)      

        del(train_loader)
        train_loss_list.append(train_loss)


        if val_iter is not None:
            if epoch%val_iter == val_iter - 1 or epoch == 0:
                if val_sample:
                    # val_surf_vars, val_vol_vars, val_surfs, val_vols = [], [], [], []
                    for i in range(20):
                        val_dataset_sampled = []
                        for data in val_dataset:
                            data_sampled = data.clone()
                            # idx = random.sample(range(data_sampled.x.size(0)), hparams['subsampling'])
                            # idx = torch.tensor(idx)
                            # data_sampled.pos = data_sampled.pos[idx].to(device)
                            # data_sampled.x = data_sampled.x[idx].to(device)
                            # data_sampled.y = data_sampled.y[idx].to(device)
                            
                            if name_mod != 'PointNet' and name_mod != 'MLP':
                                data_sampled.edge_index = nng.radius_graph(x = data_sampled.pos.to(device), r = hparams['r'], loop = True, max_num_neighbors = int(hparams['max_neighbors'])).cpu()
                            
                            val_dataset_sampled.append(data_sampled)
                        val_loader = DataLoader(val_dataset_sampled, batch_size = 1, shuffle = True)
                        del(val_dataset_sampled)
                        val_outs, val_loss, _ = test(device, model, val_loader, final_epoch, criterion)
                        del(val_loader)
                else:
                    val_outs, val_loss, _ = test(device, model, val_loader, final_epoch, criterion)
                val_epochs.append(pbar_train)
                train_loss_list.append(train_loss)
                val_loss_list.append(val_loss)

                pbar_train.set_postfix(train_loss = train_loss, val_loss = val_loss)
            else:
                pbar_train.set_postfix(train_loss = train_loss, val_loss = val_loss)
        else:
            pbar_train.set_postfix(train_loss = train_loss)
    
    train_loss_list = np.array(train_loss_list)
    val_loss_list = np.array(val_loss_list)

    end = time.time()
    time_elapsed = end - start
    params_model = get_nb_trainable_params(model).astype('float')
    print('Number of parameters:', params_model)
    print('Time elapsed: {0:.2f} seconds'.format(time_elapsed))
    print('              {0:.2f} minutes'.format(time_elapsed/60))
    print('              {0:.2f} hours'.format(time_elapsed/3600))
    torch.save(model, osp.join(path, 'model'))


    ######################################
    #                                    #
    #              Outputs               #
    #                                    #
    ######################################
    sns.set()
    fig_train_surf, ax_train_surf = plt.subplots(figsize = (20, 5))
    ax_train_surf.plot(train_loss_list, label = 'Training loss')
    ax_train_surf.set_xlabel('epochs')
    ax_train_surf.set_yscale('log')
    ax_train_surf.set_title('Train Losses:  ' + criterion)
    ax_train_surf.legend(loc = 'best')
    fig_train_surf.savefig(osp.join(path, f'{criterion}_train_loss.png'), dpi = 150, bbox_inches = 'tight')

    fig_train_vol, ax_train_vol = plt.subplots(figsize = (20, 5))
    x_axis = [i * 10 for i in range(len(val_loss_list))]
    ax_train_vol.plot(x_axis, val_loss_list, label='Validation loss')
    ax_train_vol.set_xlabel('epochs')
    ax_train_vol.set_yscale('log')
    ax_train_vol.set_title('Val Losses:  ' + criterion)
    ax_train_vol.legend(loc = 'best')
    fig_train_vol.savefig(osp.join(path, f'{criterion}_val_loss.png'), dpi = 150, bbox_inches = 'tight')
    
    fig_both, ax_both = plt.subplots(figsize = (20, 5))
    ax_both.plot(train_loss_list, label = 'Training loss')
    ax_train_vol.plot(x_axis, val_loss_list, label='Validation loss')
    ax_both.set_xlabel('epochs')
    ax_both.set_yscale('log')
    ax_both.set_title('Train Losses:  ' + criterion)
    ax_both.legend(loc = 'best')
    fig_both.savefig(osp.join(path, f'{criterion}_both_losses.png'), dpi = 150, bbox_inches = 'tight')
    
    print('Graphs Saved')

    # if val_iter is not None:
    #     fig_val_surf, ax_val_surf = plt.subplots(figsize = (20, 5))
    #     ax_val_surf.plot(val_surf_list, label = 'Mean loss')
    #     ax_val_surf.plot(val_surf_var_list[:, 0], label = r'$v_x$ loss'); ax_val_surf.plot(val_surf_var_list[:, 1], label = r'$v_y$ loss')
    #     ax_val_surf.plot(val_surf_var_list[:, 2], label = r'$p$ loss'); ax_val_surf.plot(val_surf_var_list[:, 3], label = r'$\nu_t$ loss')
    #     ax_val_surf.set_xlabel('epochs')
    #     ax_val_surf.set_yscale('log')
    #     ax_val_surf.set_title('Validation losses over the surface')
    #     ax_val_surf.legend(loc = 'best')
    #     fig_val_surf.savefig(osp.join(path, 'val_loss_surf.png'), dpi = 150, bbox_inches = 'tight')

    #     fig_val_vol, ax_val_vol = plt.subplots(figsize = (20, 5))
    #     ax_val_vol.plot(val_vol_list, label = 'Mean loss')
    #     ax_val_vol.plot(val_vol_var_list[:, 0], label = r'$v_x$ loss'); ax_val_vol.plot(val_vol_var_list[:, 1], label = r'$v_y$ loss')
    #     ax_val_vol.plot(val_vol_var_list[:, 2], label = r'$p$ loss'); ax_val_vol.plot(val_vol_var_list[:, 3], label = r'$\nu_t$ loss')
    #     ax_val_vol.set_xlabel('epochs')
    #     ax_val_vol.set_yscale('log')
    #     ax_val_vol.set_title('Validation losses over the volume')
    #     ax_val_vol.legend(loc = 'best')
    #     fig_val_vol.savefig(osp.join(path, 'val_loss_vol.png'), dpi = 150, bbox_inches = 'tight');
        
    if val_iter is not None:
        with open(osp.join(path, name_mod + '_log.json'), 'a') as f:
            json.dump(
                {
                    'regression': 'Total',
                    'loss': 'MSE',
                    'nb_parameters': params_model,
                    'time_elapsed': time_elapsed,
                    'hparams': hparams,
                    'train_loss': train_loss_list[-1],
                    # 'train_loss_surf_var': loss_surf_var_list[-1],
                    # 'train_loss_vol': train_loss_vol_list[-1],
                    # 'train_loss_vol_var': loss_vol_var_list[-1],
                    # 'val_loss_surf': val_surf_list[-1],
                    # 'val_loss_surf_var': val_surf_var_list[-1],
                    # 'val_loss_vol': val_vol_list[-1],
                    'val_loss': val_loss_list[-1],
                }, f, indent = 12, cls = NumpyEncoder
            )

    return model, val_outs