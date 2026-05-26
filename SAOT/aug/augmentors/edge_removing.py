from aug.augmentors.augmentor import Graph, Augmentor
from aug.augmentors.functional import dropout_adj


class EdgePerturbation(Augmentor):
    def __init__(self, pe: float):
        super(EdgePerturbation, self).__init__()
        self.pe = pe

    def augment(self, g: Graph) -> Graph:
        if self.pe == 0:
            return g
        x, edge_index, edge_weights, _ = g.unfold()
        edge_index, edge_weights = dropout_adj(edge_index, edge_attr=edge_weights, p=self.pe)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights, subset=None)
