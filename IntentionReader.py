"""

Using a previously trained memory of the scenario, this class manages the decoding of testing examples.
TODO: incremental, progressive computation (this version does everything in batch)

"""

from Learner import Learner
from Intention import Intention
from SkeletonAcquisitor import SkeletonAcquisitor
import itertools


class IntentionReader(SkeletonAcquisitor):
    def __init__(self, environment):
        super().__init__()              # Base class initializer
        self.dataset2d = []             # 2-D dataset
        self.intentions = []            # Intentions
        if isinstance(environment, Learner):
            self.env = environment      # Learner object representing the learned environment
        else:
            print("[ERROR] Environment must be an instance of Learner class")
            quit(-1)

    # Processes skeleton and intention data
    def initialize(self, path):
        self.generate_skeletons(path)
        self.generate_dataset()
        self.do_pca()
        self.generate_intentions()

    # Performs dimensionality reduction from 20-D to 2-D through PCA using the pre-trained model
    def do_pca(self):
        self.dataset2d = self.env.pca.transform(self.dataset).tolist()

    # Computes intentions for each training sequence
    def generate_intentions(self):
        # Consider every sequence
        previous = 0
        for offset_index in range(len(self.offsets)):
            intention = Intention()
            for data_index in range(previous, self.offsets[offset_index]):
                # Retrieve the cluster id
                cluster_id = self.env.find_closest_centroid(self.dataset2d[data_index])
                if len(intention.actions) == 0 or intention.actions[-1] != cluster_id:
                    intention.actions.append(cluster_id)
            # No goal must be specified in testing phase
            # Save the computed intention
            self.intentions.append(intention)
            previous = self.offsets[offset_index]

    # Generate a testing dataset
    def make_testing_dataset(self):
        return list(itertools.chain.from_iterable(intention.actions for intention in self.intentions))
