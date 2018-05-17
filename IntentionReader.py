"""

Using a previously trained memory of the scenario, this class manages the decoding of testing examples. This is done in
real-time, clustering new skeleton data as it arrives and analyzing the cluster transitions.

"""

from Learner import Learner
from Intention import Intention
from StopThread import StopThread
from Skeleton import NoHumansFoundException
import numpy as np
from asyncio import QueueFull


class IntentionReader(StopThread):
    def __init__(self, robot, transition_queue):
        StopThread.__init__(self)
        self.skeletons = []  # Observed skeletons
        self.offsets = []  # Splits the dataset in sequences
        self.dataset = []  # 20-D dataset
        self.dataset2d = []  # 2-D dataset
        self.intention = Intention()
        self.robot = robot
        self.transition_queue = transition_queue    # Event queue read by the upper level
        self.env = None     # Learner object representing the learned environment

    # Sets a working environment
    def set_environment(self, environment):
        if isinstance(environment, Learner):
            self.env = environment
        else:
            print("[ERROR] Environment must be an instance of Learner class")
            quit(-1)

    # Concurrently observes the scene and extracts skeletons
    def run(self):
        if self.env is None:
            print("[ERROR] Environment must be initialized!")
            quit(-1)
        self.stop_flag = False  # This is done to avoid unexpected behavior
        print("[DEBUG] IntentionReader thread is running in background.")
        while not self.stop_flag:
            try:
                # Tries to extract a skeleton
                skeleton = self.robot.look_for_skeleton()
                self.skeletons.append(skeleton)
                # Converts that skeleton to a feature array and memorizes it
                feature = skeleton.as_feature()
                if not self.dataset:
                    self.dataset = feature
                else:
                    self.dataset = np.vstack((self.dataset, skeleton.as_feature()))
                # Applies PCA
                feature2d = self.env.pca.transform(feature).tolist()
                if not self.dataset2d:
                    self.dataset2d = feature2d
                else:
                    self.dataset2d = np.vstack((self.dataset2d, feature2d))
                # Cluster and examine the transitions
                cluster_id = self.env.find_closest_centroid(feature2d)
                if len(self.intention.actions) == 0 or self.intention.actions[-1] != cluster_id:
                    self.intention.actions.append(cluster_id)       # No goal must be specified in testing phase
                    # Notify the new transition to the upper level
                    try:
                        self.transition_queue.put(cluster_id)
                    except QueueFull:
                        print("[ERROR] Transition_queue item is full and cannot accept further insertions.")
            except NoHumansFoundException:
                pass
        print("[DEBUG] Shutting down IntentionReader thread.")
