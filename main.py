import argparse, yaml, os, json, glob
import torch
import train
# from dataset import Dataset
import os.path as osp
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

from normalise import normalise, denormalise, denormalise_ys, fit
from dataset import Dataset
from data.visualise_error import error_graphs
from data.data_loader import data_loader


parser = argparse.ArgumentParser()
parser.add_argument('model', help = 'The model you want to train, choose between MLP, GraphSAGE, PointNet, GUNet', type = str)
parser.add_argument('-n', '--nmodel', help = 'Number of trained models for standard deviation estimation (default: 1)', default = 1, type = int)
parser.add_argument('-w', '--weight', help = 'Weight in front of the surface loss (default: 1)', default = 1, type = float)
parser.add_argument('-t', '--task', help = 'Task to train on. Choose between "full", "scarce", "reynolds" and "aoa" (default: full)', default = 'full', type = str)
parser.add_argument('-f', '--foils', help = 'Number of foils to train on', default = 5, type = int)
parser.add_argument('-e', '--epochs', help = 'Number of epochs to train over', default = 400, type = int)
args = parser.parse_args()


t_split = 0.7
num_foils = args.foils

work_path='E:/turb_model/Re_3M'
laptop_fp = '../50_foils'

location = 'laptop'
#location = 'work'
if location == 'laptop':
    file_path = laptop_fp
elif location == 'work':
    file_path = work_path

print('-----------------------------------------------')
print( f'Loading {num_foils} airfoils')
print('-----------------------------------------------')
# val_set = 0
if location == 'laptop':
    set_list = random.sample(range(num_foils), num_foils)
else:
    set_list = random.sample(range(1770), num_foils)

train_set = set_list[:round(len(set_list) * t_split)]
val_set = set_list[round(len(set_list) * t_split):]
d_set = []
train_dataset = []
val_dataset = []


for foil in tqdm(train_set, desc="Loading Training Data"):
    for alf in range(24):
        data = data_loader(foil, alf, file_path=file_path)
        train_dataset.append(data)

for foil in tqdm(val_set, desc = "Loading Validation Data"):
    for alf in range(24):
        data = data_loader(foil, alf, file_path=file_path)
        val_dataset.append(data)

coef_norm = fit(train_dataset)

train_dataset = normalise(train_dataset, coef_norm)
val_dataset  = normalise(val_dataset, coef_norm)

# # print(train_dataset[0])
# print(train_dataset[0].y)
# quit()



print('-----------------------------------------------')
print('-----------------------------------------------')
print( 'Running: '+ args.model + f'             for {num_foils} airfoils')
print('-----------------------------------------------')
# Cuda
use_cuda = torch.cuda.is_available()
device = 'cuda:0' if use_cuda else 'cpu'
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


plt.close()
