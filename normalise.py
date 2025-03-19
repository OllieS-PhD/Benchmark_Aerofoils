import numpy as np
import torch



def normalise(dataset, coeff_norm=None):
    if coeff_norm is None:
        # Umean = np.linalg.norm(data.x[:, 2:4], axis = 1).mean()     
        for k, data in enumerate(dataset):
            init = data.x
            target = data.y
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
                old_length = data.x.shape[0]
                std_in = ((data.x - mean_in)**2).sum(axis = 0, dtype = np.double)/old_length
                std_out = ((data.y - mean_out)**2).sum(axis = 0, dtype = np.double)/old_length
            else:
                new_length = old_length + data.x.shape[0]
                std_in += (((data.x - mean_in)**2).sum(axis = 0, dtype = np.double) - data.x.shape[0]*std_in)/new_length
                std_out += (((data.y - mean_out)**2).sum(axis = 0, dtype = np.double) - data.x.shape[0]*std_out)/new_length
                old_length = new_length
        
        std_in = np.sqrt(std_in).astype(np.single)
        std_out = np.sqrt(std_out).astype(np.single)
    else:
        (mean_in, std_in, mean_out, std_out) = coef_norm 

    # Normalize
    for data in dataset:
        data.x = torch.tensor((data.x - mean_in)/(std_in + 1e-8)).to(torch.float32)
        data.y = torch.tensor((data.y - mean_out)/(std_out + 1e-8)).to(torch.float32)
        # print(data.x[2:3,:])
        # quit()
    print(mean_in, std_in, mean_out, std_out)
    coef_norm = [mean_in, std_in, mean_out, std_out]        
    return dataset, coeff_norm

def denormalise(dataset, coef_norm):
    [mean_in, std_in, mean_out, std_out] = coef_norm 
    for data in dataset:
        data.x = ( data.x.cpu().numpy() * (std_in+1e-8) ) + mean_in
        data.y = ( data.y.cpu().numpy() * (std_out+1e-8) ) + mean_out
    
    return dataset

def denormalise_ys(dataset, coef_norm):
    (mean_in, std_in, mean_out, std_out) = coef_norm 
    for data in dataset:
        data.x = ( data.x * (std_out+1e-8) ) + mean_out
        data.y = ( data.y * (std_out+1e-8) ) + mean_out
    
    return dataset