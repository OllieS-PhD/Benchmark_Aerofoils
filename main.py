import argparse, yaml, os, json, glob
import torch
import train, metrics
# from dataset import Dataset
import os.path as osp
import numpy as np
from tqdm import tqdm
import time
import copy

from normalise import normalise, denormalise, denormalise_ys, fit
from dataset import Dataset
from data.visualise_error import error_graphs
from data.post_process import post_process
from data.data_loader import data_loader
# print('torch version:       '+torch.__version__)


parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-s', '--score', help = 'If you want to compute the score of the models on the associated test set. (default: 0)', default = 0, type = int)
parser.add_argument('-f', '--foils', help = 'Number of foils to train on', default = 5, type = int)
parser.add_argument('-e', '--epochs', help = 'Number of epochs to train over', default = 400, type = int)
args = parser.parse_args()


#################################
#       AIRFRANS DATA           #
#################################
# with open('E:/AirfRANS_Data/AirfRANS-main/Dataset/manifest.json', 'r') as f:
#     manifest = json.load(f)

# manifest_train = manifest[args.task + '_train']
# test_dataset = manifest[args.task + '_test'] if args.task != 'scarce' else manifest['full_test']
# n = int(.1*len(manifest_train))
# train_dataset = manifest_train[:-n]
# val_dataset = manifest_train[-n:]

# train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)

# val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)




'''
USING OWN DATA HERE, OVERFITTING ON 20 foils
When doing a proper go at it, randomise all 1830 into 2 groups of 70:30 
'''

from pathlib import Path


t_split = 0.7
num_foils = args.foils

print('-----------------------------------------------')
print('-----------------------------------------------')
print( 'Running: '+ args.model + f'             for {num_foils} airfoils')
print('-----------------------------------------------')

# val_set = 0
val_set = range((int(num_foils*t_split)), num_foils)
d_set = []
train_dataset = []
val_dataset = []

for foil in tqdm(range(int(num_foils*t_split)), desc="Loading Training Data"):
    for alf in range(24):
        data = data_loader(foil, alf)
        train_dataset.append(data)

for foil in tqdm(val_set, desc = "Loading Validation Data"):
    for alf in range(24):
        data = data_loader(foil, alf)
        val_dataset.append(data)
# for alf in range(24):
#     data = data_loader(4, alf)
#     train_dataset.append(data)
#     val_dataset.append(data)
# norm_set.append(data)


coef_norm = fit(train_dataset)
# print(coef_norm[0])
# print(coef_norm[1])
# print(coef_norm[2])
# print(coef_norm[3])
# quit()
# print(train_dataset[0].y)

train_dataset = normalise(train_dataset, coef_norm)
val_dataset  = normalise(val_dataset, coef_norm)

# # print(train_dataset[0])
# print(train_dataset[0].y)
# quit()



# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
print('-----------------------------------------------')
if use_cuda:
    print('Using GPU')
else:
    print('Using CPU')

with open('params.yaml', 'r') as f: # hyperparameters of the model
    hparams = yaml.safe_load(f)[args.model]


from models.MLP import MLP
models = []
for i in range(args.nmodel):
    encoder = MLP(channel_list=hparams['encoder'], batch_norm = False)
    decoder = MLP(channel_list=hparams['decoder'], batch_norm = False)

    if args.model == 'GraphSAGE':
        from models.GraphSAGE import GraphSAGE
        model = GraphSAGE(hparams, encoder, decoder)
    
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        model = PointNet(hparams, encoder, decoder)

    elif args.model == 'MLP':
        from models.NN import NN
        model = NN(hparams, encoder, decoder)

    if args.model == 'GUNet':
        from models.GUNet import GUNet
        model = GUNet(hparams, encoder, decoder)   
    
    num_epochs=args.epochs
    log_path = osp.join('metrics', f'{str(num_foils)}_foils', f'{str(num_epochs)}_epochs' ,args.model) # path where you want to save log and figures    
    model, val_outs_norm = train.main(device, train_dataset, val_dataset, model, hparams, log_path, coef_norm, 
                criterion = 'MSE', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True, num_epochs=num_epochs)
    models.append(model)
torch.save(models, osp.join(log_path, args.model))

proc_tik = time.time()
print('-----------------------------------------------')


val_outs_denorm = copy.deepcopy(val_outs_norm)
val_outs_denorm = denormalise_ys(val_outs_denorm, coef_norm)

# post_process(val_outs_denorm, args.model, hparams, num_foils,'de_norm')
# post_process(val_outs_norm, args.model, hparams, num_foils,'norm')
print(f'Done:   {(time.time()-proc_tik)/60} mins')

# print('-----------------------------------------------')
# print('Creating Error Graphs...')
# err_tik = time.time()
# rmse_list = []
# dn_rmse_list = []
# proc_vars = ['rho_u', 'rho_v', 'rho_mag', 'e', 'omega']
# types = ['x', 'y', 'err']
# tqdm_err = tqdm(val_set)

# for foil_n in tqdm_err:
#     for var in proc_vars:
#         for alpha in range(24):
#             for type in types:
#                 tqdm_err.set_postfix(foil=foil_n, alpha=alpha)
#                 err = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'norm', type=type)
#                 err_dnorm = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'de_norm', type=type)
#                 if types == 'err':
#                     rmse_list.append(err)
#                     dn_rmse_list.append(err_dnorm)
# rmse_total = sum(rmse_list) / len(rmse_list)
# dn_rmse_total = sum(dn_rmse_list) / len(dn_rmse_list)
# print('-----------------------------------------------')
# print(f'Normalised RMSE Total =     {rmse_total}')
# print(f'De-Normalised RMSE Total =  {dn_rmse_total}')
# print(f'Normalisation Coefs:        {coef_norm}')
# foil_n = 4
# alpha = 10
# for alpha in range(24):
#     for type in types:
#         for var in tqdm_err:
#             err = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'norm', type=type)
#             err_dnorm = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'de_norm', type =type )

if bool(args.score):
    s = args.task + '_test' if args.task != 'scarce' else 'full_test'
    coefs = metrics.Results_test(device, [models], [hparams], coef_norm, path_in = 'Dataset', path_out = 'scores', n_test = 3, criterion = 'MSE', s = s)
    # models can be a stack of the same model (for example MLP) on the task s, if you have another stack of another model (for example GraphSAGE)
    # you can put in model argument [models_MLP, models_GraphSAGE] and it will output the results for both models (mean and std) in an ordered array.
    np.save(osp.join('scores', args.task, 'true_coefs'), coefs[0])
    np.save(osp.join('scores', args.task, 'pred_coefs_mean'), coefs[1])
    np.save(osp.join('scores', args.task, 'pred_coefs_std'), coefs[2])
    for n, file in enumerate(coefs[3]):
        np.save(osp.join('scores', args.task, 'true_surf_coefs_' + str(n)), file)
    for n, file in enumerate(coefs[4]):
        np.save(osp.join('scores', args.task, 'surf_coefs_' + str(n)), file)
    np.save(osp.join('scores', args.task, 'true_bls'), coefs[5])
    np.save(osp.join('scores', args.task, 'bls'), coefs[6])
