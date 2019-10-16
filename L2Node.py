"""

Data needed for Progressive Clustering computation on a L2 node.

"""

from sklearn.decomposition import PCA


class L2Node:
    def __init__(self, id, data):
        self.id = id                # id is the parent id, not the subcluster id
        self.dataset = data
        self.dataset2d = None
        self.pca = None
        # Calculates PCA
        self.do_pca()

    # Performs dimensionality reduction from n-D to 2-D through PCA
    def do_pca(self):
        # PCA to reduce dimensionality to 2D
        self.pca = PCA(n_components=2).fit(self.dataset)
        self.dataset2d = self.pca.transform(self.dataset).tolist()

    # Retrieves data needed for progressive clustering computation
    def get_data(self):
        return self.dataset2d, self.pca
