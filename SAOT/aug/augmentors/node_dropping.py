from aug.augmentors.augmentor import Graph, Augmentor
from aug.augmentors.functional import drop_node
from torch_geometric.utils import to_networkx, to_dense_adj

from aug.augmentors.dropout import dropout_node

class NodeDropping(Augmentor):
    def __init__(self, pn: float):
        super(NodeDropping, self).__init__()
        self.pn = pn

    def augment(self, g: Graph) -> Graph:
        if self.pn == 0:
            return g
        
        x, edge_index, edge_weights, _ = g.unfold()
        
        # adj1 = to_dense_adj(edge_index=edge_index)

        # edge_index, edge_mask, node_mask = dropout_node(edge_index, num_nodes=x.shape[0])

        subset, edge_index, edge_weights = drop_node(edge_index, edge_weights, keep_prob=1. - self.pn)

        # adj2 = to_dense_adj(edge_index=edge_index)
        # x = x[subset,:]

        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights, subset=subset)
