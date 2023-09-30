import torch
import os

def gpu_cpu(params):
    if params == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            print('Device set to CUDA')
        else:
            device = 'cpu'
            print('Cuda device not found, device set to CPU')
    if params == 'cpu':
        device = 'cpu'
        print('Device set to CPU')
        
    return device


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
