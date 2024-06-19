import torch
from torch import nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, resnet50
from PIL import Image
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import argparse
from util import gpu_cpu, mkdir
from natsort import natsorted

class roi_dataset(Dataset):
    """
    Simple Dataset class to load the tiles for feature extraction
    """
    def __init__(self, images_list, trnsfrms_val):
        super().__init__()
        self.transform = trnsfrms_val()
        self.images_list = images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        path = self.images_list[idx]
        image = Image.open(path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


def simclr_histo_features_extractor(MODEL_PATH, RETURN_PREACTIVATION=True):
    """
    Loading the simclr model, pretrained on histological images (script and model taken from https://github.com/ozanciga/self-supervised-histopathology)
    """
    #RETURN_PREACTIVATION = True #return features from the model, if false return classification logits
    NUM_CLASSES = 1  # only used if RETURN_PREACTIVATION = False
    
    def load_model_weights(model, weights):
    
        model_dict = model.state_dict()
        weights = {k: v for k, v in weights.items() if k in model_dict}
        if weights == {}:
            print('No weight could be loaded..')
        model_dict.update(weights)
        model.load_state_dict(model_dict)
    
        return model
    
    model = torchvision.models.__dict__['resnet18'](weights=None)
    state = torch.load(MODEL_PATH, map_location='cuda:0')
    
    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    
    model = load_model_weights(model, state_dict)
    
    if RETURN_PREACTIVATION:
        model.fc = torch.nn.Sequential()
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
        
    return model


def pretrained_resnet_features_extractor(model_version):
    """
    Loading a resnet18 model, pretrained on imagenet, without the classification head, for feature extraction
    """
    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()
            
        def forward(self, x):
            return x
    
    if model_version == 'resnet18':
        model = resnet18(weights="IMAGENET1K_V1")
    if model_version == 'resnet50':
        model = resnet50(weights="IMAGENET1K_V2")
    model.fc = Identity()
    
    return model


def trnsfrms_val():
    """
    Transformations applied to each patch before feature extraction
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsfrm = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    return trnsfrm


def batch_names(name_list, batch_size):
    """
    Batching for feature extraction
    """
    batched_names = []
    lower_bound = 0
    while lower_bound+batch_size <= len(name_list):
        batched_names.append(name_list[lower_bound:lower_bound+batch_size])
        lower_bound += batch_size
    
    #handle the last batch separatly if the remaining data is smaller than the batch size
    last_batch = name_list[lower_bound:lower_bound+batch_size]
    if len(last_batch) > 0:
        batched_names.append(last_batch)
        
    return batched_names


def features_extraction(model, patches_path, extracted_features_path, batch_size):
    """
    Feature extraction and saving files
    """
    paths_array = []
    patches_paths = glob(os.path.join(patches_path, '*'))
    for patch in tqdm(patches_paths):
        for level in glob(os.path.join(patch, '*')):
            if os.path.isdir(level):
                for img_path in glob(os.path.join(level, '*')):
                    paths_array.append(img_path)

    extracted_features_path = 'features'
    array_names = []
    for path in paths_array:
        path_sep = path.split(os.path.sep)
        f1 = os.path.join(extracted_features_path, path_sep[1])
        mkdir(f1)
        f2 = os.path.join(f1, path_sep[2])
        mkdir(f2)
        new_path = os.path.join(f2, path_sep[3]).split('.')[0] + '.npy'
        array_names.append(new_path)

    array_names = natsorted(array_names)
    batched_names = batch_names(array_names, batch_size)
    dataset = roi_dataset(paths_array, trnsfrms_val)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print("Extracting features...")

    with torch.no_grad():
        for name_batch, batch in tqdm(zip(batched_names, data_loader)):
            batch = batch.to(device)
            features = model(batch)
            features = features.cpu().numpy()
            for name, feature_vector in zip(name_batch, features):
                np.save(name, feature_vector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', dest='model', default='simclr')
    parser.add_argument('-p', dest='model_path', default='tenpercent_resnet18.ckpt')
    parser.add_argument('-s', dest='source')
    parser.add_argument('-d', dest='destination')
    parser.add_argument('-b', dest='batch_size', default=256)
    parser.add_argument('-c', dest='device', default='cuda')
    params = parser.parse_args()
    
    if params.model == 'simclr':
        model = simclr_histo_features_extractor(params.model_path)
    elif 'resnet' in params.model:
        model = pretrained_resnet_features_extractor(params.model)
    
    device = gpu_cpu(params.device)
    
    model = model.to(device)
    model.eval()
    
    features_extraction(model, params.source, params.destination, int(params.batch_size))
    
