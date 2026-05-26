from aug.augmentors.augmentor import Graph, Augmentor
from aug.augmentors.functional import drop_feature


class FeatureMasking(Augmentor):
    def __init__(self, pf: float):
        super(FeatureMasking, self).__init__()
        self.pf = pf

    def augment(self, g: Graph) -> Graph:
        if self.pf == 0:
            return g
        x, edge_index, edge_weights, _ = g.unfold()
        x = drop_feature(x, self.pf)
        return Graph(x=x, edge_index=edge_index, edge_weights=edge_weights, subset= None)
