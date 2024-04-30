import torch
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torch.nn import Linear, Dropout
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.dense import dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
"""
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.25):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.norm1 = LayerNorm(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.norm2 = LayerNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        self.norm3 = LayerNorm(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)

        self.norm4 = LayerNorm(hidden_channels)
        self.norm5 = LayerNorm(hidden_channels)

        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionalAggregation(gate_nn=Linear(hidden_channels, 1))

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.norm1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.dropout(x)

        x = self.norm2(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()
        x = self.dropout(x)

        x = self.norm3(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()
        x = self.dropout(x)

        x = self.norm4(x)
        x = self.attention_pooling(x, batch)
        x = self.dropout(x)

        x = self.norm5(x)
        x = self.lin1(x).relu()
        x = self.dropout(x)

        x = self.lin2(x)

        return torch.sigmoid(x).squeeze()

    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.25):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.norm1 = LayerNorm(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.norm2 = LayerNorm(hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)

        self.dropout = Dropout(dropout_rate)
        self.attention_pooling = AttentionalAggregation(gate_nn=Linear(hidden_channels, 1))

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.norm1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.dropout(x)

        x = self.norm2(x)
        x = self.attention_pooling(x, batch)
        x = self.dropout(x)

        x = self.norm3(x)
        x = self.lin1(x).relu()
        x = self.dropout(x)

        x = self.lin2(x)

        return torch.sigmoid(x).squeeze()
""" 
    
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, criterion, dropout_rate=0.25):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        
        self.criterion = criterion

        self.norm1 = LayerNorm(in_channels)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)

        self.norm2 = LayerNorm(hidden_channels)
        self.norm3 = LayerNorm(hidden_channels)
        
        self.pool = Linear(hidden_channels, hidden_channels)

        self.dropout = Dropout(dropout_rate)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch, true_labels=None):
        
        x = self.norm1(x)
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        x = self.dropout(x)
        
        #TODO: apply mask to not count padding values in average computing
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        s = self.pool(x)
        x, adj, mincut_loss, orthogonality_loss = dense_mincut_pool(x, adj, s, mask)
        x = self.norm2(x)
        
        x = x.mean(dim=1)#TODO: replace this with a ViT
        
        x = self.lin2(x)
        
        out = torch.sigmoid(x).squeeze()
        if true_labels is not None:
            loss = self.criterion(out, true_labels) + mincut_loss + orthogonality_loss
            return out, loss
        return out
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    