import argparse, yaml, os, json, glob
import torch
import train, metrics
# from dataset import Dataset
import os.path as osp
import numpy as np
from tqdm import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

from normalise import normalise, denormalise, denormalise_ys
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

# # if os.path.exists('Dataset/train_dataset'):
# #     train_dataset = torch.load('Dataset/train_dataset')
# #     val_dataset = torch.load('Dataset/val_dataset')
# #     coef_norm = torch.load('Dataset/normalization')
# # else:
# train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)
# # torch.save(train_dataset, 'Dataset/train_dataset')
# # torch.save(coef_norm, 'Dataset/normalization')
# val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)

# # if os.path.exists('Dataset/train_dataset'):
# #     train_dataset = torch.load('Dataset/train_dataset')
# #     val_dataset = torch.load('Dataset/val_dataset')
# #     coef_norm = torch.load('Dataset/normalization')
# # else:
# # train_dataset, coef_norm = Dataset(train_dataset, norm = True, sample = None)
# # # torch.save(train_dataset, 'Dataset/train_dataset')
# # # torch.save(coef_norm, 'Dataset/normalization')
# # val_dataset = Dataset(val_dataset, sample = None, coef_norm = coef_norm)
# # torch.save(val_dataset, 'Dataset/val_dataset')



'''
USING OWN DATA HERE, OVERFITTING ON 20 foils
When doing a proper go at it, randomise all 1830 into 2 groups of 70:30 
'''

t_split = 0.8
num_foils = 5
# x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
x_scaler, y_scaler = StandardScaler(), StandardScaler()

print('-----------------------------------------------')
print( 'Running: '+ args.model + f'             for {num_foils} airfoils')
print('-----------------------------------------------')

val_set = range((int(num_foils*t_split)), num_foils)
# val_set = 0
coef_norm = None
d_set = []
train_dataset = []
val_dataset = []
esp = 1e-10

# d_init = data_loader(0,7)
# set_scale_x = [[1.0, 0.1, 0.1, esp, esp, max(d_init.x[5,:])],
#             [0, -0.1, -0.1, -esp, -esp, min(d_init.x[5,:])]]
# set_scale_y = [[1.0, 0.1, 0.1, esp, esp],
#             [0, -0.1, -0.1, -esp, -esp]]

# x_scaler.fit(set_scale_x)
# y_scaler.fit(set_scale_y)
# print(f'pre: {d_init.x=}')
# d_init.x = x_scaler.transform(d_init.x)
# print(f'post: {d_init.x=}')
# quit()
# print(f'pre: {d_init.y=}')
# d_init.y = y_scaler.fit_transform(d_init.y)
# print(f'post: {d_init.y=}')
# quit()

# scaler.fit(d_init.y)

for foil in tqdm(range(int(num_foils*t_split)), desc="Loading Training Data"):
    for alf in range(24):
        data = data_loader(foil, alf)
        train_dataset.append(data)
for foil in tqdm(val_set, desc = "Loading Validation Data"):
    for alf in range(24):
        data = data_loader(foil, alf)
        val_dataset.append(data)

train_dataset, coeff_norm = normalise(train_dataset)
val_dataset, _ = normalise(val_dataset, coeff_norm)
print(coeff_norm)

# foil_n = 0
# alpha = 10
# data = data_loader(0,alpha)
# # torch.set_printoptions(threshold=float('inf'))
# # print(f'pre {data.x=}')
# # print(f'pre {data.y=}')
# scaler.fit(data.y)
# data.x = torch.tensor(scaler.transform(data.x)).to(torch.float32)
# data.y = torch.tensor(scaler.transform(data.y)).to(torch.float32)
# # print(f'{data.x=}')
# # print(f'{data.y=}')
# train_dataset.append(data)
# val_dataset.append(data)
# # quit()



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
    
    num_epochs=hparams['nb_epochs'] 
    log_path = osp.join('metrics', f'{str(num_foils)}_foils', f'{str(num_epochs)}_epochs' ,args.task, args.model) # path where you want to save log and figures    
    model, val_outs = train.main(device, train_dataset, val_dataset, model, hparams, log_path, 
                criterion = 'MSE', val_iter = 10, reg = args.weight, name_mod = args.model, val_sample = True)
    models.append(model)
torch.save(models, osp.join(log_path, args.model))


proc_tik = time.time()
print('-----------------------------------------------')
# still_norm = val_outs
val_outs = denormalise_ys(val_outs, coeff_norm)
post_process(val_outs, args.model, hparams, num_foils,'de_norm')
# post_process(still_norm, args.model, hparams, num_foils,'norm')
print(f'Done:   {(time.time()-proc_tik)/60} mins')

print('-----------------------------------------------')
print('Creating Error Graphs...')
err_tik = time.time()
rmse_list = []
dn_rmse_list = []
proc_vars = ['rho_u', 'rho_v', 'rho_mag', 'e', 'omega']
tqdm_err = tqdm(proc_vars)
# for var in tqdm_err:
#     for foil_n in val_set:
#         for alpha in range(24):
#             tqdm_err.set_postfix(foil=foil_n, alpha=alpha)
#             err = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'norm')
#             err_dnorm = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'de_norm')
#             rmse_list.append(err)
#             dn_rmse_list.append(err_dnorm)
# rmse_total = sum(rmse_list) / len(rmse_list)
# dn_rmse_total = sum(dn_rmse_list) / len(dn_rmse_list)
# print('-----------------------------------------------')
# # print(f'Normalised RMSE Total = {rmse_total}')
# print(f'De-Normalised RMSE Total = {dn_rmse_total}')
foil_n = 4
alpha = 10
for var in tqdm_err:
    err = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'norm')
    err_dnorm = error_graphs(foil_n, alpha, num_foils, num_epochs, args.model, var, folder = 'de_norm')

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
