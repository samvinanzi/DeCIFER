"""

Using a previously trained memory of the scenario, this class manages the decoding of testing examples. This is done in
real-time, clustering new skeleton data as it arrives and analyzing the cluster transitions.

"""

from Learner import Learner
from Intention import Intention
from Skeleton import NoHumansFoundException
import numpy as np
from iCub import icub
import time


class IntentionReader:
    def __init__(self, transition_queue):
        self.skeletons = []  # Observed skeletons
        self.offsets = []  # Splits the dataset in sequences
        self.dataset = None  # 20-D dataset
        self.dataset2d = None  # 2-D dataset
        self.intention = Intention()
        self.tq = transition_queue    # Event queue read by the upper level
        self.env = None     # Learner object representing the learned environment

    # Sets a working environment
    def set_environment(self, environment):
        assert isinstance(environment, Learner), "Environment must be an instance of Learner class"
        self.env = environment

    # Observes the scene and reads intentions
    def observe(self, fps=2):
        assert self.env is not None, "Environment must be initialized"
        image_containers = icub.initialize_yarp_image()
        print("[DEBUG] " + self.__class__.__name__ + " is observing")
        i = 0
        goal_found = False
        while not goal_found:
            try:
                # Tries to extract a skeleton
                skeleton = icub.look_for_skeleton(image_containers, i)
                self.skeletons.append(skeleton)
                # Converts that skeleton to a feature array and memorizes it
                feature = skeleton.as_feature()
                # It is a single sample, so reshape it
                feature = feature.reshape(1, -1)
                if self.dataset is None:
                    self.dataset = feature
                else:
                    self.dataset = np.vstack((self.dataset, skeleton.as_feature()))
                # Applies PCA
                feature2d = self.env.pca.transform(feature).tolist()
                if self.dataset2d is None:
                    self.dataset2d = feature2d
                else:
                    self.dataset2d = np.vstack((self.dataset2d, feature2d))
                # Cluster and examine the transitions
                cluster_id = self.env.find_closest_centroid(*feature2d)     # Unpacking of the list
                if len(self.intention.actions) == 0 or self.intention.actions[-1] != cluster_id:
                    self.intention.actions.append(cluster_id)       # No goal must be specified in testing phase
                    # Notify the new transition to the upper level
                    self.tq.put(cluster_id)
                    print("[DEBUG][IR] Wrote " + str(cluster_id) + " to transition queue")
                i += 1
            except NoHumansFoundException:
                pass
            finally:
                if self.tq.was_goal_inferred():     # Checks to see if a goal was found to decide if to stop processing
                    goal_found = True
                else:
                    time.sleep(1 / fps)
        print("[DEBUG] " + self.__class__.__name__ + " stopped observing")      # currently unreachable
