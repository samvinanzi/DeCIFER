"""

Using a previously trained memory of the scenario, this class manages the decoding of testing examples.

"""

from Learner import Learner
from Intention import Intention
import itertools
from threading import Thread
from Skeleton import NoHumansFoundException


class IntentionReader(Thread):
    def __init__(self, robot, environment=None):
        Thread.__init__(self)

        # From SkeletonAcquisitor...
        self.skeletons = []  # Observed skeletons
        self.offsets = []  # Splits the dataset in sequences
        self.dataset = []  # 20-D dataset

        self.stop_flag = False
        self.robot = robot

        self.dataset2d = []             # 2-D dataset
        self.intentions = []            # Intentions
        if environment is not None:
            self.set_environment(environment)
        else:
            self.env = environment

    # Concurrently observes the scene and extracts skeletons
    def run(self):
        while not self.stop_flag:
            try:
                skeleton = self.robot.look_for_skeleton()
                # Converts that skeleton to a feature array
                feature = skeleton.as_feature()
                # Applies PCA
                feature2d = self.env.pca.transform(feature).tolist()
                # ToDo generate intentions AND remember globally the past findings
            except NoHumansFoundException:
                pass

    # Sets a working environment
    def set_environment(self, environment):
        if isinstance(environment, Learner):
            self.env = environment      # Learner object representing the learned environment
        else:
            print("[ERROR] Environment must be an instance of Learner class")
            quit(-1)

    # Processes skeleton and intention data
    def initialize(self, path):
        #self.generate_skeletons(path)
        #self.generate_dataset()
        self.do_pca()
        self.generate_intentions()

    # Performs dimensionality reduction from 20-D to 2-D through PCA using the pre-trained model
    def do_pca(self):
        if self.env is None:
            print("[ERROR] Environment not initialized.")
            quit(-1)
        self.dataset2d = self.env.pca.transform(self.dataset).tolist()

    # Computes intentions for each training sequence
    def generate_intentions(self):
        if self.env is None:
            print("[ERROR] Environment not initialized.")
            quit(-1)
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
