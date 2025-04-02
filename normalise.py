import numpy as np
import torch

def fit(dataset):
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()     
    for k, data in enumerate(dataset):
        init = np.copy(data.x)
        target = np.copy(data.y)
        if k == 0:
            old_length = init.shape[0]
            mean_in = init.mean(axis = 0, dtype = np.double)
            mean_out = target.mean(axis = 0, dtype = np.double)
        else:
            new_length = old_length + init.shape[0]
            mean_in += (init.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_in)/new_length
            mean_out += (target.sum(axis = 0, dtype = np.double) - init.shape[0]*mean_out)/new_length
            old_length = new_length 
        # Compute normalization
        mean_in = mean_in.astype(np.single)
        mean_out = mean_out.astype(np.single)
        # data.x = data.x/torch.tensor([6, 6, Umean, Umean, 6, 1, 1], dtype = torch.float)
        # data.y = data.y/torch.tensor([Umean, Umean, .5*Umean**2, Umean], dtype = torch.float)

        if k == 0:
            old_length = init.shape[0]
            std_in = ((init - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
            std_out = ((target - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
        else:
            new_length = old_length + init.shape[0]
            std_in += (((init - mean_in)**2).sum(axis = 0, dtype = np.double) - init.shape[0]*std_in)/new_length
            std_out += (((target - mean_out)**2).sum(axis = 0, dtype = np.double) - target.shape[0]*std_out)/new_length
            old_length = new_length
    
    std_in = np.sqrt(std_in).astype(np.single)
    std_out = np.sqrt(std_out).astype(np.single)
    coef_norm = [mean_in, std_in, mean_out, std_out]     
    return coef_norm


def normalise(dataset, coeff_norm):
    mean_in, std_in, mean_out, std_out = coeff_norm
    # Normalize
    for data in dataset:
        data.x = torch.tensor((np.array(data.x) - mean_in)/(std_in + 1e-8)).to(torch.float32)
        data.y = torch.tensor((np.array(data.y) - mean_out)/(std_out + 1e-8)).to(torch.float32)
        # print(data.x[2:3,:])
        # quit()       
    return dataset

def denormalise(dataset, coef_norm):
    mean_in, std_in, mean_out, std_out = coef_norm 
    for data in dataset:
        # data = data.cpu()
        data.x = torch.tensor(( np.array(data.x) * (std_in+1e-8) ) + mean_in).to(torch.float32)
        data.y = torch.tensor(( np.array(data.y) * (std_out+1e-8) ) + mean_out).to(torch.float32)
    
    return dataset

def denormalise_ys(dataset, coef_norm):
    mean_in, std_in, mean_out, std_out = coef_norm 
    for l_data in dataset:
        data = l_data[0].cpu()
        # data = data.cpu()
        data.x = torch.tensor(( np.array(data.x) * (std_out+1e-8) ) + mean_out).to(torch.float32)
        data.y = torch.tensor(( np.array(data.y) * (std_out+1e-8) ) + mean_out).to(torch.float32)
    
    return dataset