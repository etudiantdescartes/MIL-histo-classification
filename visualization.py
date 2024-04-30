import torch
from glob import glob
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def get_coordinates(graph, scale_factor=None):
    coords = graph.metadata['coordinates']
    pos = dict()
    for i, coord in enumerate(coords):
        pos[i] = (coord[0] + coord[2]) // 2, (coord[1] + coord[3]) // (-2)
        
    if scale_factor is not None:
        pos = {node: (int(x * scale_factor), int(y * scale_factor)) for node, (x, y) in pos.items()}
    
    return pos
        
def display_graph(graph):
    pos = get_coordinates(graph)
    edges = graph.edge_index.numpy().T
    G = nx.Graph()
    G.add_edges_from(edges)
    nx_draw(G, pos, None)
    
def nx_draw(G, pos, color):
    plt.figure(figsize=(20,20))
    nx.draw(G, pos=pos, with_labels=False, node_size=10)
    plt.show()


def attention_map(graph, model, self_loops=False):#TODO: finish this function
    edges, attention_weights = model(graph.x, graph.edge_index, return_attention_weights=True)[1]#TODO: handle mulitple attention layers
    edges = edges.numpy().T
    attention_weights = attention_weights.squeeze().detach().numpy()
    pos = get_coordinates(graph, 100)
    self_loops_edges_mask = edges[:,0] == edges[:,1]
    
    if self_loops:
        e = edges[self_loops_edges_mask]
        a = attention_weights[self_loops_edges_mask]
    
    else:
        e = edges[~self_loops_edges_mask]
        a = attention_weights[~self_loops_edges_mask]
        G = nx.DiGraph()
        G.add_edges_from(e)
        
        fig, ax = plt.subplots(figsize=(400,400))
        
        nx.draw_networkx_nodes(G, pos=pos, node_size=150, node_color='lightblue')
        nx.draw_networkx_edges(G, pos=pos, width=3, edge_cmap=plt.get_cmap('viridis'), edge_color=a, connectionstyle='arc3,rad=0.1', arrowsize=4)
        
        norm = Normalize(vmin=a.min(), vmax=a.max())
        mappable = ScalarMappable(norm=norm, cmap=plt.get_cmap('viridis'))
        plt.colorbar(mappable, ax=ax, label='Edge importance')
        ax.set_aspect('equal')
        plt.savefig('attention_graph.png')  
        plt.show()
    
    
    
#TODO: find a way to represent attention regions to perform IOU thresholding with annotations
#TODO: compare them at different magnification scales
#TODO: optionally, display attention weights from the attentionalAggregation layer


def graph_visualization(graph, model, display_attention=False):
    pass



if __name__ == '__main__':
    graph_list = glob('pyg_graphs/*.pt')
    graph = torch.load('pyg_graphs/tumor_028.pt')
    model_path = 'model/best_model.pt'
    model = torch.load(model_path).to('cpu')
    model.eval()

    attention_map(graph, model, False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    