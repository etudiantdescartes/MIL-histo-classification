from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random
import networkx as nx
from torch_geometric.utils import to_networkx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import argparse
from glob import glob
from util import gpu_cpu


from model import GCN


class GraphDataset(Dataset):
    """
    Simple Dataset class to format the graph list
    """
    def __init__(self, graph_list, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(None, transform, pre_transform)
        self.graph_list = graph_list

    def len(self):
        return len(self.graph_list)

    def get(self, idx):
        return self.graph_list[idx]


def train(loader, model, optimizer, criterion):
    """
    Training over one epoch
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        data.edge_index = data.edge_index.to(torch.int64)
        out, loss = model(data.x, data.edge_index, data.batch, data.y.squeeze().float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validation(loader, model, optimizer, criterion):
    """
    Training over one epoch
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.edge_index = data.edge_index.to(torch.int64)
            out, loss = model(data.x, data.edge_index, data.batch, data.y.squeeze().float())
            total_loss += loss.item()
    return total_loss / len(loader)



def plot_training_curves(train_curve, val_curve):
    plt.plot(train_curve, label='Train loss', color='blue')
    plt.plot(val_curve, label='Val loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss curves')
    plt.legend()
    plt.show()


def delete_metadata(dataset):
    """
    Deleting metadata from train val sets to reduce the size of the graph objects and speed up training
    """
    for data in tqdm(dataset):
        if 'metadata' in data:
            del data.metadata

"""
def train_loop(train_loader, val_loader, model, optimizer, criterion, num_epochs, use_validation=True):
    model = model.to(device)
    train_curve = []
    val_curve = []
    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, optimizer, criterion)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        train_curve.append(train_loss)

        if use_validation:
            val_loss = validation(val_loader, model, optimizer, criterion)
            print(f'Epoch: {epoch:03d}, Val Loss: {val_loss:.4f}')
            val_curve.append(val_loss)

    plot_training_curves(train_curve, val_curve)
"""

def train_loop(train_loader, val_loader, model, optimizer, criterion, num_epochs):
    model = model.to(device)
    train_curve = []
    val_curve = []
    min_val_loss = np.inf
    for epoch in range(num_epochs):
        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss = validation(val_loader, model, optimizer, criterion)
        val_accuracy = accuracy(val_loader, model)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        train_curve.append(train_loss)
        val_curve.append(val_loss)

        if val_loss < min_val_loss:
            print('Model saved')
            min_val_loss = val_loss
            torch.save(model, 'model/best_model.pt')

    plot_training_curves(train_curve, val_curve)


def accuracy(loader, model, training=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            preds = (out >= 0.5).float()
            if not training:
                print(f'preds: {preds}')
                print(f'true: {data.y}')
                
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    accuracy = correct / total
    return accuracy


def auroc(loader, model):
    model.eval()
    ground_truths = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            if out.dim() == 0:
                predictions.append(out.cpu().item())
                ground_truths.append(data.y.cpu().item())
            else:
                predictions.extend(out.cpu().numpy())
                ground_truths.extend(data.y.cpu().numpy())

    auroc = roc_auc_score(ground_truths, predictions)
    #TODO: add ROC figures for each fold
    return auroc



def k_fold_cross_validation(X, k):
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    y = [graph.y for graph in X]
    total_auroc = 0

    criterion = torch.nn.BCELoss()
    num_epochs = 2

    for fold in k_fold.split(X, y):

        model = GCN(in_channels=512, hidden_channels=1024, out_channels=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

        train_list = [X[ind] for ind in fold[0]]
        test_list = [X[ind] for ind in fold[1]]

        train_set = GraphDataset(train_list)
        test_set = GraphDataset(test_list)

        train_loader = DataLoader(train_set, batch_size=4, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

        train_loop(train_loader, model, optimizer, criterion, num_epochs)
        total_auroc += auroc(test_loader, model)

    mean_auroc = total_auroc / k

    return mean_auroc



def split_train_val_test(graphs, train=0.7, test=0.15, val=0.15, balance_dataset=False):

    def split(graphs):
        train_size = int(train * len(graphs))
        val_size = int(val * len(graphs))
        train_set = graphs[:train_size]
        val_set = graphs[train_size:train_size+val_size]
        test_set = graphs[train_size+val_size:]
        return train_set, val_set, test_set

    negative_class = [graph for graph in graphs if graph.y==0]
    positive_class = [graph for graph in graphs if graph.y==1]

    if balance_dataset:
        print(f'Negatives: {len(negative_class)}, Positives: {len(positive_class)}')

        if len(positive_class) > len(negative_class):
            positive_class = positive_class[:len(negative_class)]
        else:
            negative_class = negative_class[:len(positive_class)]

    print(f'Negatives: {len(negative_class)}, Positives: {len(positive_class)}')

    train_set, val_set, test_set = [GraphDataset(a+b) for a, b in zip(split(negative_class), split(positive_class))]

    return train_set, val_set, test_set


def class_weights(dataset, device):
    negative = 0
    positive = 0
    for graph in dataset:
        if graph.y == 0:
            negative += 1
        else:
            positive += 1

    postive_weight = torch.tensor([negative / positive], dtype=torch.float).to(device)
    return postive_weight
    

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='source', default='pyg_graphs')
    parser.add_argument('-d', dest='destination')#TODO: save conf matrices etc.
    parser.add_argument('-c', dest='device', default='cuda')
    parser.add_argument('-k', dest='k', default=5)
    parser.add_argument('-b', dest='batch_size', default=1)
    #TODO: config file and argparse to have both options
    params = parser.parse_args()
    
    graph_list = glob(f'{params.source}/*.pt')
    device = gpu_cpu(params.device)

    graphs = [torch.load(graph) for graph in graph_list]

    delete_metadata(graphs)
    random.shuffle(graphs)
    train_set, val_set, test_set = split_train_val_test(graphs)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False)

    out_channels = 1
    pos_weight = class_weights(train_set, device)
    print(f'Positive weight: {pos_weight.item():.3f}')
    criterion = torch.nn.BCELoss(pos_weight)
    model = GCN(512, 512, 1, criterion)
    weight_decay = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=weight_decay)

    
    num_epochs = 20
    
    train_loop(train_loader, val_loader, model, optimizer, criterion, num_epochs)
    
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
    best_model = torch.load('model/best_model.pt')
    best_model = best_model.to(device)
    print(f'Accuracy: {accuracy(test_loader, best_model, False):.3f}')

    """
    mean_auroc = k_fold_cross_validation(graph_list, k)
    print(f'Mean auroc score using {k}-fold validation')
    """
