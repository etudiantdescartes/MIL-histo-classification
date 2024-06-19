import numpy as np
import os
import torch
from glob import glob
from torch_geometric.data import Data, Dataset
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm


class GraphDataset(Dataset):
    """
    Simple Dataset class for the graph data
    """
    def __init__(self, graph_list, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]
    

def parse_tile_names(tile_path):
    """
    Parse tile names to get the tile coordinates within its original WSI
    Returns a dictionary containing the coordinates and path (name) of the tile
    """
    name = os.path.basename(tile_path).split('.')[0]
    split_name = name.split('_')
    x1, y1, x2, y2 = [int(split_name[i]) for i in range(len(split_name)) if i > 2]
    level = split_name[2].split('level')[1]
    
    return {'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'level': level, 'path': tile_path}


def in_radius(tile_a, tile_b, radius):
    """
    Checks if the distance between two tiles is within the specified radius
    """
    xa = (float(tile_a['x2']) + float(tile_a['x1'])) / 2
    ya = (float(tile_a['y2']) + float(tile_a['y1'])) / 2
    
    xb = (float(tile_b['x2']) + float(tile_b['x1'])) / 2
    yb = (float(tile_b['y2']) + float(tile_b['y1'])) / 2
    
    distance = np.sqrt((xa-xb)**2 + (ya-yb)**2)
    
    if distance <= radius:
        return True
    
    return False
    

def graph_creation(tile_list, annotations):
    """
    Returns a pytorch_geometric Data object (graph)
    """
    if len(tile_list) == 0:
        return None
    else:

        edges = []
        for i, tile_a in enumerate(tile_list):
            for j, tile_b in enumerate(tile_list[i+1:], start=i+1):
                if in_radius(tile_a, tile_b, radius=640):#320 for one-hop 8 connected graph
                    edges.append((i,j))
                    edges.append((j,i))
        edges = torch.tensor(edges)

        node_features = torch.tensor(np.array([np.load(tile['path']) for tile in tile_list]))
        
        wsi_name = os.path.basename(tile_list[0]['path']).split('_', 2)
        wsi_name = wsi_name[0] + '_' + wsi_name[1] + '.xml'
        target = [1 if wsi_name in annotations else 0]
        target = torch.tensor(target)
        
        graph = Data(x=node_features, edge_index=edges.t().contiguous(), y=target)
        
    coords = [(tile['x1'], tile['y1'], tile['x2'], tile['y2']) for tile in tile_list]
    graph.metadata = {'name': wsi_name.split('.')[0], 'coordinates': coords}
    return graph


def process(path_list, annotations, destination):
    """
    Single process function to create one graph (easier for multiprocessing)
    """
    tiles = [parse_tile_names(tile_path) for tile_path in path_list]
    graph = graph_creation(tiles, annotations)
    graph_name = graph['metadata']['name']
    torch.save(graph, f'{destination}/{graph_name}.pt')
    

def bundle_patches(patch_paths):
    """
    Bundles patches from the same WSI together, since the path list contains every path
    """
    bundled_patches = {}
    for patch_path in patch_paths:
        wsi_name = os.path.basename(patch_path).split('_', 2)
        wsi_name = wsi_name[0] + '_' + wsi_name[1]
        
        if wsi_name in bundled_patches:
            bundled_patches[wsi_name].append(patch_path)
        else:
            bundled_patches[wsi_name] = [patch_path]
    
    return list(bundled_patches.values())


def remove_none_graphs(graph_list):
    """
    Util function to remove None graphs if there are any
    """
    none_count = 0
    while None in graph_list:
        graph_list.remove(None)
        none_count += 1
    print(f'{none_count} None graphs were removed')



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='source', default='features')
    parser.add_argument('-d', dest='destination', default='pyg_graphs')
    parser.add_argument('-p', dest='num_processes', default=8)
    params = parser.parse_args()

    #features_paths = glob(f'{params.source}/*.npy')
    features_paths = []
    for root, dirs, files in os.walk(params.source):
        for file in files:
            features_paths.append(os.path.join(root, file))

    
    bundled = bundle_patches(features_paths)

    annotations = [os.path.basename(annot) for annot in glob('annotations/*.xml')]
    
    cpu_cores = int(params.num_processes)
    print(f'Number of processes: {cpu_cores}')
    print('Creating graphs...')
    arg_list = [(bundle, annotations, params.destination) for bundle in bundled]
    with mp.Pool(cpu_cores) as p:
        p.starmap(process, arg_list)
        
    """
    #Iterative processing for debugging
    graph_list = []
    for bundle in tqdm(bundled):
        process(bundle, annotations, params.destination)
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
        