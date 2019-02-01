"""

This class is in charge of learning intentions from training examples, managing everything that goes from skeleton
extraction to cluster generation.

"""

from Skeleton import Skeleton
from Cluster import Cluster
from Intention import Intention
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pyclustering.cluster.xmeans as pyc
from pyclustering.utils import draw_clusters
import os
import pickle
import csv
import math
from iCub import icub


class Learner:
    dimensions = 2              # Currently working with 2D skeletons

    def __init__(self, debug=False):
        self.skeletons = []     # Observed skeletons
        self.offsets = []       # Splits the dataset in sequences
        self.dataset = []       # 20-D dataset
        self.dataset2d = []     # 2-D dataset
        self.clusters = []      # Clusters
        self.intentions = []    # Intentions
        self.goal_labels = []   # Goal labels
        self.pca = None         # Trained parameters of a PCA model
        self.ax = None          # Plotting purpose
        self.debug = debug      # If true, vocal commands will be substituted by keyboard input

    # --- INITIALIZATION METHODS --- #

    # Acquires and processes the training data
    # Can be a new, fresh learning or the update of an existing knowledge base
    def learn(self, new=True, savedir="objects/"):
        if new:
            self.generate_skeletons()
        else:
            self.update_knowledge()
        self.generate_dataset()
        self.do_pca()
        self.generate_clusters()
        self.generate_intentions()
        if savedir is not None:
            self.save(savedir)

    # Reload already computed Controller data
    def reload_data(self, path="objects/"):
        try:
            self.load(path)
        except Exception:
            print("Error: failed to load Controller data.")
            quit(-1)

    # --- SAVE AND LOAD METHODS --- #

    # Saves the objects in binary format
    def save(self, savedir="objects/"):
        if savedir[-1] != '/':
            savedir += '/'
        # Verify the existence of the saving directory
        dir = os.path.dirname(savedir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Saves each attribute of the class
        pickle.dump(self.skeletons, open(savedir + "skeletons.p", "wb"))
        pickle.dump(self.dataset, open(savedir + "dataset.p", "wb"))
        pickle.dump(self.dataset2d, open(savedir + "dataset2d.p", "wb"))
        pickle.dump(self.clusters, open(savedir + "clusters.p", "wb"))
        pickle.dump(self.offsets, open(savedir + "offsets.p", "wb"))
        pickle.dump(self.intentions, open(savedir + "intentions.p", "wb"))
        pickle.dump(self.goal_labels, open(savedir + "goal_labels.p", "wb"))
        pickle.dump(self.pca, open(savedir + "pca.p", "wb"))
        pickle.dump(self.ax, open(savedir + "ax.p", "wb"))

    # Loads the objects from binary format
    def load(self, path="objects/"):
        if path[-1] != '/':
            path += '/'
        # Loads each attribute of the class
        self.skeletons = pickle.load(open(path + "skeletons.p", "rb"))
        self.dataset = pickle.load(open(path + "dataset.p", "rb"))
        self.dataset = pickle.load(open(path + "dataset.p", "rb"))
        self.dataset2d = pickle.load(open(path + "dataset2d.p", "rb"))
        self.clusters = pickle.load(open(path + "clusters.p", "rb"))
        self.offsets = pickle.load(open(path + "offsets.p", "rb"))
        self.intentions = pickle.load(open(path + "intentions.p", "rb"))
        self.goal_labels = pickle.load(open(path + "goal_labels.p", "rb"))
        self.pca = pickle.load(open(path + "pca.p", "rb"))
        self.ax = pickle.load(open(path + "ax.p", "rb"))

    # --- METHODS --- #

    # Performs the learning by demonstration task
    def generate_skeletons(self):
        # Loops for all the goals to be learned
        finished = False
        i = 0
        while not finished:
            icub.say("Please, start demonstrating.")
            # Learns a single goal
            skeletons, goal_name = icub.record_goal(i, fps=2, debug=self.debug)
            self.skeletons.extend(skeletons)    # extend instead of append to avoid nesting lists
            self.goal_labels.append(goal_name)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
            i += len(skeletons)
            icub.say("Do you want to show me another goal?")
            while True:
                if not self.debug:
                    response = icub.wait_and_listen()
                else:
                    response = icub.wait_and_listen_dummy()
                if icub.recognize_commands(response, listenFor="NO"):
                    icub.say("All right, thanks for showing me.")
                    finished = True
                    break
                elif icub.recognize_commands(response, listenFor="YES"):
                    icub.say("Ok then, let's continue.")
                    break
                else:
                    icub.say("Sorry, I didn't understand. Can you repeat?")

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self):
        # Creates the dataset array
        dataset = np.zeros(shape=(1, 10*Learner.dimensions))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature()))
        # Removes the first, empty row
        self.dataset = dataset[1:]

    # Performs dimensionality reduction from 20-D to 2-D through PCA
    def do_pca(self):
        # PCA to reduce dimensionality to 2D
        self.pca = PCA(n_components=2).fit(self.dataset)
        self.dataset2d = self.pca.transform(self.dataset).tolist()

    # Performs X-Means clustering on the provided dataset
    def generate_clusters(self):
        # initial centers with K-Means++ method
        initial_centers = kmeans_plusplus_initializer(list(self.dataset2d), 2).initialize()
        # create object of X-Means algorithm that uses CCORE for processing
        # Default tolerance: 0.025
        xmeans_instance = pyc.xmeans(self.dataset2d, initial_centers, ccore=True, kmax=50, tolerance=0.025,
                                     criterion=pyc.splitting_type.BAYESIAN_INFORMATION_CRITERION)
        # run cluster analysis
        xmeans_instance.process()
        # obtain results of clustering
        centers = xmeans_instance.get_centers()
        cluster_lists = xmeans_instance.get_clusters()
        colors = ['red', 'blue', 'darkgreen', 'brown', 'violet', 'deepskyblue', 'darkgrey', 'lightsalmon', 'deeppink',
                  'yellow', 'black', 'mediumspringgreen', 'orange', 'darkviolet', 'darkblue', 'silver', 'lime', 'pink',
                  'gold', 'bisque']
        # Sanity check
        if len(centers) > len(colors):
            print("[WARNING] More than " + str(len(colors)) + " clusters detected, cannot display them all.")
            # Extend the colors list to avoid index out of bounds during next phase
            # Visualization will be impaired but the computation will not be invalidated
            colors.extend(['white'] * (len(centers) - len(colors)))
        for i in range(len(centers)):
            c = Cluster(centers[i], cluster_lists[i], i, colors[i])
            self.clusters.append(c)
        # generate plot and optionally display it
        self.ax = draw_clusters(self.dataset2d, cluster_lists, display_result=False)

    # Find the cluster id containing the skeleton (make it more pythonic)
    def find_cluster_id(self, skeleton_id):
        for i in range(len(self.clusters)):
            if skeleton_id in self.clusters[i].skeleton_ids:
                return i
        return None

    # Saves a dataset as a csv
    def save_csv(self):
        file = open('csv/dataset.csv', 'w')
        with file:
            writer = csv.writer(file)
            writer.writerows(self.dataset)

    # Finds the closest centroid to a new skeletal sample (already in 2D coordinates)
    def find_closest_centroid(self, sample2d):
        min_distance = float("inf")
        closest_cluster = None
        for cluster in self.clusters:
            dist = math.hypot(sample2d[0] - cluster.centroid[0], sample2d[1] - cluster.centroid[1])
            if dist < min_distance:
                min_distance = dist
                closest_cluster = cluster.id
        return closest_cluster

    # Computes intentions for each training sequence
    def generate_intentions(self):
        # Checks that lengths are equals
        if len(self.goal_labels) != len(self.offsets):
            print("[ERROR] Dimension mismatch between goal label list and offset list.")
            quit(-1)
        # Consider every sequence
        previous = 0
        for offset_index in range(len(self.offsets)):
            intention = Intention()
            for skeleton_index in range(previous, self.offsets[offset_index]):
                # Retrieve the cluster id
                cluster_id = self.find_cluster_id(self.skeletons[skeleton_index].id)
                if len(intention.actions) == 0 or intention.actions[-1] != cluster_id:
                    intention.actions.append(cluster_id)
            # Create the goal label from pathname
            intention.goal = self.goal_labels[offset_index]
            # Save the computed intention
            self.intentions.append(intention)
            previous = self.offsets[offset_index]

    # Generates a list containing the intentions in dictionary form (training dataset)
    def make_training_dataset(self):
        dict_list = []
        for intention in self.intentions:
            dict_list.append(intention.as_dict())
        return dict_list

    # Determines the general reach orientation of each cluster
    def cluster_orientation_reach(self):
        output = [''] * len(self.clusters)
        for cluster in self.clusters:
            orientations = {
                "left": 0,
                "right": 0,
                "center": 0
            }
            for skeleton_id in cluster.skeleton_ids:
                orientations[self.skeletons[skeleton_id].orientation_reach()] += 1
            output[cluster.id] = max(orientations)
        return output

    # Updates the knowledge base with a new goal
    # If the name of the goal is already know, replaces the previous training received
    def update_knowledge(self):
        # Fetch the last id of the dataset
        last_id = max([skeleton.id for skeleton in self.skeletons])
        # Learn one new goal
        print("[DEBUG] Acquiring a new goal...")
        new_skeletons, new_goal = icub.record_goal(last_id+1, fps=2, debug=self.debug)
        # Check to see if the goal is a new one or a replacement
        if new_goal in self.goal_labels:
            # Get the id of the old goal which is being replacing
            discard_id = self.goal_labels.index(new_goal)
            self.dataset_shift(discard_id, new_skeletons)
        else:
            self.skeletons.extend(new_skeletons)  # extend instead of append to avoid nesting lists
            self.goal_labels.append(new_goal)
            self.offsets.append(len(new_skeletons) + max(self.offsets))
        # Re-train the system
        self.learn(new=False, savedir=None)

    # Replaces a skeleton dataset for a specified goal. The existing skeletons associated to the training of that task
    # are eliminated, the new ones are appended to the set and all the ids are shifted left to guarantee continuity
    def dataset_shift(self, discard_id, new_set):
        # Sanity checks
        assert 0 <= discard_id <= len(self.offsets)-1, "[ERROR] discard_id must be a valid index of the offsets list."
        assert all(isinstance(n, Skeleton) for n in new_set), "[ERROR] new_set must contain Skeleton types."
        # Delete the deprecated elements from the dataset
        discard_offset = self.offsets[discard_id]
        if discard_id == 0:
            offset_begin = 0
        else:
            offset_begin = self.offsets[discard_id - 1]
        for x in range(offset_begin, discard_offset):
            self.skeletons[x] = None
        self.skeletons = [x for x in self.skeletons if x is not None]
        # Extend with the new data
        self.skeletons.extend(new_set)
        # Shift back the ids of the elements
        for i in range(len(self.skeletons)):
            if self.skeletons[i].id != i:
                self.skeletons[i].id = i
        # Fix the offsets
        start = 0
        for i in range(len(self.offsets)):      # Offsets to subset lengths
            temp = self.offsets[i]
            self.offsets[i] -= start
            start = temp
        self.offsets.extend([len(new_set)])     # Addition of the new subset length
        self.offsets.pop(discard_id)            # Removal of the unused subset length
        start = 0
        for i in range(len(self.offsets)):      # Subset lengths to offsets
            self.offsets[i] += start
            start = self.offsets[i]

    # --- DISPLAY METHODS --- #

    # Displays a human-friendly result of the clustering operation
    def show_clustering(self, just_dots=False):
        # Sanity check
        if self.ax is None:
            print("Error: must generate clusters before trying to display them.")
            raise RuntimeError
        if not just_dots:
            # Create interactive plot
            for skeleton in self.skeletons:
                im = OffsetImage(skeleton.img, zoom=0.38)
                coordinates = self.dataset2d[skeleton.id]
                cluster_id = self.find_cluster_id(skeleton.id)
                # Sanity check
                if cluster_id is None:
                    print("Error! Couldn't find cluster in which a skeleton belongs")
                    raise RuntimeError
                else:
                    color = self.clusters[cluster_id].color
                    ab = AnnotationBbox(im, coordinates, bboxprops=dict(edgecolor=color))
                    self.ax.add_artist(ab)
                    # ax.text(coordinates[0]+0.0015, coordinates[1]+0.005, skeleton.id, fontsize=25)
        plt.show()

    # Plots the clusters centroids
    def plot_clusters(self):
        x = []
        y = []
        for cluster in self.clusters:
            x.append(cluster.centroid[0])
            y.append(cluster.centroid[1])
            self.ax.text(cluster.centroid[0] + 0.0015, cluster.centroid[1] + 0.005, cluster.id, fontsize=25)
        plt.plot(x, y, 'kD')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cluster Centroids')
        plt.grid(True)
        plt.show()

    # Plots the whole action representations for each goal, with the associated skeletons
    def plot_goal(self):
        # Flush previous plots
        plt.clf()
        plt.cla()
        plt.close()

        last_id = 0
        skeleton_list = []
        for goal in self.intentions:
            skeleton_count = 0
            print("\nGoal: " + goal.goal + " " + str(goal.actions))
            # Phase 1: obtain the skeletons
            for cluster_id in goal.actions:
                sk_id = [id for id in self.clusters[cluster_id].skeleton_ids if id > last_id][0]
                skeleton_list.append(sk_id)
                last_id = sk_id
                skeleton_count += 1
                print(str(sk_id) + " ")
            # Phase 2: plot
            fig = plt.figure()
            # Start plotting each subfigure
            for i in range(skeleton_count):
                skeleton = self.skeletons[skeleton_list[i]]
                # set up the axes for the i-th plot
                ax = fig.add_subplot(1, skeleton_count, i+1, projection='3d')
                ax.set_title("Cluster: " + str(goal.actions[i]) + ", Skeleton: " + str(skeleton_list[i]))
                connections = [['Head', 'Neck'],
                               ['Neck', 'Torso'],
                               ['Neck', 'RElbow'],
                               ['Neck', 'LElbow'],
                               ['RElbow', 'RWrist'],
                               ['LElbow', 'LWrist'],
                               ['Torso', 'RKnee'],
                               ['Torso', 'LKnee'],
                               ['RKnee', 'RAnkle'],
                               ['LKnee', 'LAnkle']]
                nonmissing_kp = skeleton.nonmissing_keypoints(apply_to_2d=False)
                array = skeleton.keypoints_to_array(nonmissing_kp)
                x = array[:, 0]
                y = array[:, 1]
                z = array[:, 2]
                ax.set_zlabel('Z')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                # Connect keypoints to form skeleton
                for p1, p2 in connections:
                    if p1 in nonmissing_kp and p2 in nonmissing_kp:
                        start = skeleton.keypoints[p1]
                        end = skeleton.keypoints[p2]
                        ax.plot(np.linspace(start.x, end.x), np.linspace(start.y, end.y),
                                np.linspace(start.z, end.z), c="blue", marker='.', linestyle=':', linewidth=0.1)
                # Plot the dots
                ax.scatter(x, y, z, zdir='z', c='b', marker='o', linewidths=5.0)
                plt.title('Skeleton ' + str(i) + "\nCluster: " + str(goal.actions[i]))
                plt.grid(True)
                # Puts text
                for label, keypoint in nonmissing_kp.items():
                        ax.text(keypoint.x, keypoint.y, keypoint.z, label, None)
                ax.view_init(elev=-65, azim=-90)  # Rotate the view
        plt.title(goal.goal)
        plt.show()
