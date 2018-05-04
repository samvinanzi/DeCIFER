"""

Class that models a skeleton data cluster

"""


class Cluster:
    def __init__(self, centroid, skeletons, id=0, color='red'):
        self.centroid = centroid
        self.skeleton_ids = skeletons
        self.id = id
        self.color = color

    def contains(self, id):
        return True if id in self.skeleton_ids else False
