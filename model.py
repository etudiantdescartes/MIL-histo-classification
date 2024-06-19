import torch
from torch_geometric.nn import GATConv
from torch.nn import Linear, Dropout
from torch_geometric.utils import softmax
from torch import Tensor
from typing import Optional
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.nn.inits import reset



class AttentionalAggregation(Aggregation):
    """
    Taken from the source code of pytorch_geometric to add the option to return attention scores
    Node aggregation using attention to obtain a graph representation as a single vector
    """
    def __init__(
        self,
        gate_nn: torch.nn.Module,
        nn: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        from torch_geometric.nn import MLP

        self.gate_nn = self.gate_mlp = None
        if isinstance(gate_nn, MLP):
            self.gate_mlp = gate_nn
        else:
            self.gate_nn = gate_nn

        self.nn = self.mlp = None
        if isinstance(nn, MLP):
            self.mlp = nn
        else:
            self.nn = nn

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.gate_mlp)
        reset(self.nn)
        reset(self.mlp)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:

        self.assert_two_dimensional_input(x, dim)

        if self.gate_mlp is not None:
            gate = self.gate_mlp(x, batch=index, batch_size=dim_size)
        else:
            gate = self.gate_nn(x)

        if self.mlp is not None:
            x = self.mlp(x, batch=index, batch_size=dim_size)
        elif self.nn is not None:
            x = self.nn(x)

        gate = softmax(gate, index, ptr, dim_size, dim)
        res = self.reduce(gate * x, index, ptr, dim_size, dim)
        return res, gate


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'gate_nn={self.gate_mlp or self.gate_nn}, '
                f'nn={self.mlp or self.nn})')


class GNN(torch.nn.Module):
    """
    GNN model for graph classification
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GATConv(in_channels, hidden_channels)
        self.attention_pooling = AttentionalAggregation(gate_nn=Linear(512, 1), nn=Linear(512, 512))
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = Dropout()(x)

        x, attention_scores = self.attention_pooling(x, batch)
        x = Dropout()(x)

        x = self.lin1(x).relu()
        x = Dropout()(x)
        x = self.lin2(x)

        x = torch.sigmoid(x).squeeze()
        return x, attention_scores


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    