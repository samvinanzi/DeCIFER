"""

Using a previously trained memory of the scenario, this class manages the decoding of testing examples. This is done in
real-time, clustering new skeleton data as it arrives and analyzing the cluster transitions.

"""

from Learner import Learner
from Intention import Intention
from Skeleton import Skeleton, NoHumansFoundException
import numpy as np
from robots.robot_selector import robot
import time
import subprocess
import cv2
import os


class IntentionReader:
    def __init__(self, transition_queue, logger):
        self.skeletons = []  # Observed skeletons
        self.offsets = []  # Splits the dataset in sequences
        self.dataset = None  # 20-D dataset
        self.dataset2d = None  # 2-D dataset
        self.intention = Intention()
        self.tq = transition_queue    # Event queue read by the upper level
        self.log = logger  # Logger that keeps record of what happens, for data analysis purpose
        self.env = None     # Learner object representing the learned environment

    # Sets a working environment
    def set_environment(self, environment):
        assert isinstance(environment, Learner), "Environment must be an instance of Learner class"
        self.env = environment

    # Observes the scene and reads intentions
    def observe(self, fps=2):
        assert self.env is not None, "Environment must be initialized"
        image_containers = robot.get_image_containers()
        print("[DEBUG] " + self.__class__.__name__ + " is observing")
        i = 0
        goal_found = False
        blank_detections = 0
        self.log.new_trial()
        while not goal_found:
            try:
                # Tries to extract a skeleton
                skeleton = robot.look_for_skeleton(image_containers, i)
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
                    blank_detections = 0    # reset
                    self.intention.actions.append(cluster_id)       # No goal must be specified in testing phase
                    # Notify the new transition to the upper level
                    self.tq.put(cluster_id)
                    # Increase the elapsed time in the logger
                    self.log.update_latest_time()
                    print("[DEBUG][IR] Wrote " + str(cluster_id) + " to transition queue")
                else:
                    blank_detections += 1
                    # This avoids infinite loops. If N skeletons are detected with no clust transition, then the system
                    # wasn't able to guess the intention. Manually write "unknown" in the transition queue.
                    if blank_detections >= 20:
                        print("[DEBUG] Unable to infer intentions, sorry :(")
                        self.tq.write_goal_name("failure")
                        blank_detections = 0
                i += 1
            except NoHumansFoundException:
                pass
            finally:
                # Checks to see if a goal was found to decide if to stop processing
                goal_name = self.tq.was_goal_inferred()
                if goal_name:
                    goal_found = True   # Exit condition
                    self.log.update_latest_goal(goal_name)
                elif robot.__class__.__name__ != "Sawyer":  # Sawyer doesn't need other waiting times
                    time.sleep(1 / fps)
        print("[DEBUG] " + self.__class__.__name__ + " stopped observing")

    # Offline, dataset-based observation
    def offline_observe(self, path="/home/samuele/Research/datasets/CAD-60/data1/", volume=3):
        assert self.env is not None, "Environment must be initialized"
        # Prepares the environment, pre-processing all the data
        # Load images from desired folder
        # This line of code creates an ordered list of filenames, even if the latter are not alphabetically
        # orderable (because front padding is missing, i.e. RGB_1, RGB_2 instead of RGB_1, RGB_10...)
        sorted_files = [str(path + f.decode("utf-8")) for f in
                        subprocess.check_output("ls " + path + "/ | sort -V", shell=True).splitlines()]
        # Only consideres 1/volume (default 1/3) of the data
        sorted_files = sorted_files[::volume]
        images = np.empty(len(sorted_files), dtype=object)
        for n in range(0, len(sorted_files)):
            print("Veryfing: " + str(os.path.join(path, sorted_files[n])))
            images[n] = cv2.imread(os.path.join(path, sorted_files[n]))
        # Process the skeletons one at a time
        i = 0
        id = 0
        goal_found = False
        blank_detections = 0
        self.log.new_trial()
        while not goal_found:
            try:
                # Tries to extract a skeleton
                skeleton = Skeleton(images[i], id)
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
                cluster_id = self.env.find_closest_centroid(*feature2d)  # Unpacking of the list
                if len(self.intention.actions) == 0 or self.intention.actions[-1] != cluster_id:
                    blank_detections = 0  # reset
                    self.intention.actions.append(cluster_id)  # No goal must be specified in testing phase
                    # Notify the new transition to the upper level
                    self.tq.put(cluster_id)
                    # Increase the elapsed time in the logger
                    self.log.update_latest_time()
                    print("[DEBUG][IR] Wrote " + str(cluster_id) + " to transition queue")
                else:
                    blank_detections += 1
                    # This avoids infinite loops. If 50 skeletons are detected with no clust transition, then the system
                    # wasn't able to guess the intention. Manually write "unknown" in the transition queue.
                    if blank_detections >= 20:
                        print("[DEBUG] Unable to infer intentions, sorry :(")
                        self.tq.write_goal_name("failure")
                        blank_detections = 0
                id += 1
            except NoHumansFoundException:
                pass
            finally:
                i += 1
                # Checks to see if a goal was found to decide if to stop processing
                goal_name = self.tq.was_goal_inferred()
                if goal_name:
                    goal_found = True  # Exit condition
                    self.log.update_latest_goal(goal_name)
                else:
                    time.sleep(0.5)
        print("[DEBUG] " + self.__class__.__name__ + " stopped observing")

    # Observer simulator, accepts commands from the keyboard
    def observer_simulator(self):
        assert self.env is not None, "Environment must be initialized"
        print("[DEBUG] OBSERVER SIMULATOR is online.")
        goal_found = False
        blank_detections = 0
        while not goal_found:
            cluster_id = int(input('Enter observed cluster id: '))
            if len(self.intention.actions) == 0 or self.intention.actions[-1] != cluster_id:
                blank_detections = 0  # reset
                self.intention.actions.append(cluster_id)  # No goal must be specified in testing phase
                # Notify the new transition to the upper level
                self.tq.put(cluster_id)
                print("[DEBUG][IR] Wrote " + str(cluster_id) + " to transition queue")
            else:
                blank_detections += 1
                # This avoids infinite loops. If N skeletons are detected with no clust transition, then the system
                # wasn't able to guess the intention. Manually write "unknown" in the transition queue.
                if blank_detections >= 20:
                    print("[DEBUG] Unable to infer intentions, sorry :(")
                    self.tq.write_goal_name("failure")
                    blank_detections = 0
            # Checks to see if a goal was found to decide if to stop processing
            goal_name = self.tq.was_goal_inferred()
            if goal_name:
                goal_found = True  # Exit condition
            elif robot.__class__.__name__ != "Sawyer":  # Sawyer doesn't need other waiting times
                time.sleep(0.5)
        print("[DEBUG] OBSERVER SIMLATOR offline.")
