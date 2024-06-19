from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import StratifiedKFold
from glob import glob
from scipy import interp
from sklearn.metrics import roc_curve, auc
from util import gpu_cpu
from model import GNN
import argparse


class GraphDataset(Dataset):
    """
    Simple Dataset class to format the graph list
    """
    def __init__(self, graph_list, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
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
        out, attention_scores = model(data.x, data.edge_index, data.batch)
        if out.dim() == 0:
            out = out.unsqueeze(0)
        if data.y.dim() == 2:
            target = data.y.squeeze()
        else:
            target = data.y
        loss = criterion(out, target.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validation(loader, model, optimizer, criterion):
    """
    Validation over one epoch
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            data.edge_index = data.edge_index.to(torch.int64)
            out, attention_scores = model(data.x, data.edge_index, data.batch)
            if out.dim() == 0:
                out = out.unsqueeze(0)
            if data.y.dim() == 2:
                target = data.y.squeeze()
            else:
                target = data.y
            loss = criterion(out, target.float())
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_training_curves(train_curve, val_curve):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(train_curve, label='Train loss', color='blue')
    ax.plot(val_curve, label='Val loss', color='orange')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss curves')
    ax.legend()
    plt.tight_layout()
    plt.show()


def delete_metadata(dataset):
    """
    Deleting metadata from train val sets to reduce the size of the graph objects and speed up training
    """
    for data in tqdm(dataset):
        if 'metadata' in data:
            del data.metadata
        if 'node_labels' in data:
            del data.node_labels

class EarlyStopping:
    """
    Early stopping class to stop the training when the validation loss
    doesn't decrease for a number of 'patience' epochs
    """
    def __init__(self, patience):
        self.patience = patience
        self.min_loss = np.inf
        self.counter = 0

    def early_stop(self, val_loss):
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
        elif val_loss >= self.min_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train_loop(train_loader, val_loader, model, optimizer, criterion, num_epochs, patience, model_save_path):
    model = model.to(device)
    train_curve = []
    val_curve = []
    early_stopping = EarlyStopping(patience)
    min_val_loss = np.inf
    for epoch in range(num_epochs):

        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss = validation(val_loader, model, optimizer, criterion)
        val_accuracy = accuracy(val_loader, model)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        train_curve.append(train_loss)
        val_curve.append(val_loss)

        if early_stopping.early_stop(val_loss):
            break

        if val_loss < min_val_loss:
            print('Model saved')
            min_val_loss = val_loss
            torch.save(model, f'{model_save_path}/best_model.pt')

    plot_training_curves(train_curve, val_curve)


def accuracy(loader, model, training=True):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, attn = model(data.x, data.edge_index, data.batch)
            preds = (out >= 0.5).float()
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
            out, _ = model(data.x, data.edge_index, data.batch)
            if out.dim() == 0:
                predictions.append(out.cpu().item())
                ground_truths.append(data.y.cpu().item())
            else:
                predictions.extend(out.cpu().numpy())
                ground_truths.extend(data.y.cpu().numpy())

    fpr, tpr, _ = roc_curve(ground_truths, predictions)
    roc_auc = auc(fpr, tpr)
    plot_roc_curve(fpr, tpr, roc_auc)

    return fpr, tpr, roc_auc

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def k_fold_cross_validation(X, k, batch_size, model_save_path, num_epochs, patience, lr, weight_decay):
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    y = [graph.y for graph in X]

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for fold in k_fold.split(X, y):
        train_list = [X[ind] for ind in fold[0]]
        test_list = [X[ind] for ind in fold[1]]
        random.shuffle(train_list)
        val_list = train_list[:len(test_list)]
        train_list = train_list[len(test_list):]

        train_set = GraphDataset(train_list)
        test_set = GraphDataset(test_list)
        val_set = GraphDataset(val_list)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        pos_weight = class_weights(train_set, device)
        criterion = torch.nn.BCELoss(pos_weight)

        model = GNN(512, 512, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loop(train_loader, val_loader, model, optimizer, criterion, num_epochs, patience, model_save_path)
        
        model = torch.load(f'{model_save_path}/best_model.pt')
        model = model.to(device)
        model.eval()

        fpr, tpr, roc_auc = auroc(test_loader, model)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0  
        aucs.append(roc_auc)


    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)


    plt.figure()
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.legend(loc="lower right")
    plt.show()



def split_train_val_test(graphs, train=0.7, test=0.15, val=0.15, balance_dataset=False):
    """
    Splits dataset while keeping the class balance ratio in each set
    """

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

        if len(positive_class) > len(negative_class):
            positive_class = positive_class[:len(negative_class)]
        else:
            negative_class = negative_class[:len(positive_class)]

    print(f'Negatives: {len(negative_class)}, Positives: {len(positive_class)}')
    train_set, val_set, test_set = [GraphDataset(a+b) for a, b in zip(split(negative_class), split(positive_class))]
    return train_set, val_set, test_set


def class_weights(dataset, device):
    """
    Calculate the weight to assign to positive examples
    To mitigate unbalanced dataset related training issues
    """
    negative = 0
    positive = 0

    for graph in dataset:
        if graph.y == 0:
            negative += 1
        else:
            positive += 1

    positive_weight = torch.tensor([negative / positive], dtype=torch.float).to(device)

    return positive_weight




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', dest='source')
    parser.add_argument('-d', dest='destination')
    parser.add_argument('-c', dest='device', default='cuda')
    parser.add_argument('-k', dest='k', default=5, type=int)
    parser.add_argument('-b', dest='batch_size', default=4, type=int)
    parser.add_argument('-e', dest='num_epochs', default=50, type=int)
    parser.add_argument('-p', dest='patience', default=10, type=int)
    parser.add_argument('-l', dest='lr', default=0.00005, type=float)
    parser.add_argument('-w', dest='weight_decay', default=5e-3, type=float)
    params = parser.parse_args()


    graph_list = glob(f'{params.source}/*.pt')
    device = gpu_cpu('cuda')

    graphs = [torch.load(graph) for graph in tqdm(graph_list)]

    delete_metadata(graphs)
    random.shuffle(graphs)
    
    k_fold_cross_validation(graphs, params.k, params.batch_size, params.destination, params.num_epochs, params.patience, params.lr, params.weight_decay)