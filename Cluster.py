"""

Class that models a skeleton data cluster

"""


class Cluster:
    def __init__(self, centroid, skeletons, id=0, color='red', level=1):
        self.centroid = centroid
        self.skeleton_ids = skeletons
        self.id = id
        self.color = color
        self.level = level
        self.descendants = []

    def contains(self, id):
        return True if id in self.skeleton_ids else False

    # Returns a list of colors to use for the clusters, eventually extending it (shouldn't happen)
    @staticmethod
    def get_colors(total_needed):
        colors = ['yellow', 'blue', 'red', 'lightsalmon', 'brown', 'violet', 'deepskyblue', 'darkgrey', 'deeppink',
                  'darkgreen', 'black', 'mediumspringgreen', 'orange', 'darkviolet', 'darkblue', 'silver', 'lime',
                  'pink', 'gold', 'bisque']
        # Sanity check
        if total_needed > len(colors):
            print("[WARNING] More than " + str(len(colors)) + " clusters detected, cannot display them all.")
            # Extend the colors list to avoid index out of bounds during next phase
            # Visualization will be impaired but the computation will not be invalidated
            colors.extend(['white'] * (total_needed - len(colors)))
        return colors

    # If it's a child cluster, obtain the ancestor id, otherwise return None
    def get_parent_id(self):
        id = int(self.id)
        return id if id != self.id else None

    # Is this cluster a parent?
    def is_parent(self):
        return bool(self.descendants)   # If descendants is empty, returns False
