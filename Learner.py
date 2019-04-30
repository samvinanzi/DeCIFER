"""

This class is in charge of learning intentions from training examples, managing everything that goes from skeleton
extraction to cluster generation.

"""

from Skeleton import Skeleton, NoHumansFoundException
from Cluster import Cluster
from Intention import Intention
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
from sklearn.decomposition import PCA
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pyclustering.cluster.xmeans as pyc
from pyclustering.utils import draw_clusters
from pyclustering.cluster.silhouette import silhouette
import os
import pickle
import csv
import math
#from iCub import icub
from robots.robot_selector import robot
import cv2
import subprocess
import time


class Learner:
    DIMENSIONS = 2              # Currently working with 2D skeletons
    PCA_DIMENSIONS = 2           # Dimensionality reduction target dimensions

    def __init__(self, debug=False, persist=False):
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
        self.persist = persist  # If True, saves data structures to disk

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
        score = self.generate_clusters()
        print("[DEBUG] Clustering silhouette score: " + str(score))
        self.generate_intentions()
        if self.persist:
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
            robot.say("Please, start demonstrating.")
            # Learns a single goal
            skeletons, goal_name = robot.record_goal(i, fps=2, debug=self.debug)
            self.skeletons.extend(skeletons)    # extend instead of append to avoid nesting lists
            self.goal_labels.append(goal_name)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
            i += len(skeletons)
            robot.say("Do you want to show me another goal?")
            while True:
                if not self.debug:
                    response = robot.wait_and_listen()
                else:
                    response = robot.wait_and_listen_dummy()
                if robot.recognize_commands(response, listenFor="NO"):
                    robot.say("All right, thanks for showing me.")
                    finished = True
                    break
                elif robot.recognize_commands(response, listenFor="YES"):
                    robot.say("Ok then, let's continue.")
                    break
                else:
                    robot.say("Sorry, I didn't understand. Can you repeat?")

    # Debug mode for skeleton acquisition: input from file rather than from robot eyes
    # Assumes each goal is contained in a separated subdir names as the goal itself
    def offline_learning(self, path="/home/samuele/Research/datasets/CAD-60/variations/", volume=20, savedir="objects/cad60/"):
        tic = time.time()
        i = 0
        for folder in os.listdir(path):
            print("---Processing folder: " + folder)
            basename = os.path.join(path, folder) + "/"
            # Load images from desired folder
            # This line of code creates an ordered list of filenames, even if the latter are not alphabetically
            # orderable (because front padding is missing, i.e. RGB_1, RGB_2 instead of RGB_1, RGB_10...)
            sorted_files = [str(basename + f.decode("utf-8")) for f in
                            subprocess.check_output("ls " + basename + "/ | sort -V", shell=True).splitlines()]
            # Only consideres 1/volume (default 1/5) of the data
            sorted_files = sorted_files[::volume]
            images = np.empty(len(sorted_files), dtype=object)
            for n in range(0, len(sorted_files)):
                print("Veryfing: " + str(os.path.join(basename, sorted_files[n])))
                images[n] = cv2.imread(os.path.join(basename, sorted_files[n]))
            # Create skeletons for all of them
            skeletons = []
            id = 0
            for image in images:
                try:
                    skeleton = Skeleton(image, id)
                    skeletons.append(skeleton)
                    id += 1
                except NoHumansFoundException:
                    continue
            self.skeletons.extend(skeletons)
            self.goal_labels.append(folder)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
            i += len(skeletons)
        dt = time.time() - tic
        print("\n\nProcessing completed. Elapsed time: " + str(dt / 60) + " minutes.")
        # Continue with normal learning phases
        self.generate_dataset()
        self.do_pca()
        score = self.generate_clusters()
        print("[DEBUG] Clustering silhouette score: " + str(score))
        self.generate_intentions()
        if self.persist:
            self.save(savedir)

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self):
        # Creates the dataset array
        dataset = np.zeros(shape=(1, 10*Learner.DIMENSIONS))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature()))
        # Removes the first, empty row
        self.dataset = dataset[1:]

    # Performs dimensionality reduction from 20-D to 2-D through PCA
    def do_pca(self):
        # PCA to reduce dimensionality to 2D
        self.pca = PCA(n_components=Learner.PCA_DIMENSIONS).fit(self.dataset)
        self.dataset2d = self.pca.transform(self.dataset).tolist()

    # Performs X-Means clustering on the provided dataset
    def generate_clusters(self):
        # initial centers with K-Means++ method
        initial_centers = kmeans_plusplus_initializer(list(self.dataset2d), Learner.PCA_DIMENSIONS).initialize()
        # create object of X-Means algorithm that uses CCORE for processing
        # Default tolerance: 0.025
        xmeans_instance = pyc.xmeans(self.dataset2d, initial_centers, ccore=True, kmax=20, tolerance=0.025,
                                     criterion=pyc.splitting_type.BAYESIAN_INFORMATION_CRITERION)
        # run cluster analysis
        xmeans_instance.process()
        # obtain results of clustering
        centers = xmeans_instance.get_centers()
        cluster_lists = xmeans_instance.get_clusters()
        # Calculate Silhouette score for each points and averages it
        score = np.average(silhouette(list(self.dataset2d), cluster_lists).process().get_score())
        colors = ['yellow', 'blue', 'red', 'brown', 'violet', 'deepskyblue', 'darkgrey', 'lightsalmon', 'deeppink',
                  'darkgreen', 'black', 'mediumspringgreen', 'orange', 'darkviolet', 'darkblue', 'silver', 'lime',
                  'pink', 'gold', 'bisque']
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
        return score

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

    # Summarizes the results of the training
    def training_result(self):
        print("SUMMARY OF TRAINING")
        print("Clusters: " + str(len(self.clusters)))
        print("Intentions:")
        for intention in self.intentions:
            print(str(intention.goal) + " : " + str(intention.actions))

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
            output[cluster.id] = max(orientations, key=orientations.get)    # Gets the key with highest value
        return output

    # Updates the knowledge base with a new goal
    # If the name of the goal is already know, replaces the previous training received
    def update_knowledge(self):
        # Fetch the last id of the dataset
        last_id = max([skeleton.id for skeleton in self.skeletons])
        # Learn one new goal
        print("[DEBUG] Acquiring a new goal...")
        new_skeletons, new_goal = robot.record_goal(last_id+1, fps=2, debug=self.debug)
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
            # Create image plot
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
                    ab.set_zorder(2)
                    self.ax.add_artist(ab)
        # Plots the clusters centroids
        for cluster in self.clusters:
            x = cluster.centroid[0]
            y = cluster.centroid[1]
            text = self.ax.text(x, y + 0.015, cluster.id, fontsize=40, color='white')
            # Adds the black border around the text
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
            plt.scatter(x, y, zorder=3, marker='o', s=1000, c=cluster.color, edgecolors='black', linewidths='2')
        plt.show()

    # Plots the whole action representations for each goal, with the associated skeletons
    def plot_goal(self, verbose=False):
        plt.clf()           # |
        plt.cla()           # | Flush previous plots
        plt.close()         # |
        # Font sizes
        plt.rcParams.update({'axes.titlesize': 24})
        plt.rcParams.update({'axes.labelsize': 14})
        plt.rcParams.update({'xtick.labelsize': 14})
        plt.rcParams.update({'ytick.labelsize': 14})
        last_id = -1
        skeleton_list = []
        intention_id = 0        # Used to retrieve the correct offset value for each intention
        previous_offset = 0
        for goal in self.intentions:
            skeleton_count = 0
            if verbose:
                print("\nGoal: \"" + goal.goal + "\"\nClusters: " + str(goal.actions))
            # Phase 1: obtain the skeletons
            offset = self.offsets[intention_id]
            for cluster_id in goal.actions:
                sk_id = [id for id in self.clusters[cluster_id].skeleton_ids
                         if (id > last_id and previous_offset <= id < offset)][0]
                skeleton_list.append(sk_id)
                last_id = sk_id
                skeleton_count += 1
            if verbose:
                print("Skeleton ids: " + str(skeleton_list) + "\nCount: " + str(skeleton_count))
            # Phase 2: plot
            fig = plt.figure()
            # Start plotting each subfigure
            for i in range(skeleton_count):
                skeleton = self.skeletons[skeleton_list[i]]
                # set up the axes for the i-th plot
                ax = fig.add_subplot(1, skeleton_count, i+1)
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
                nonmissing_kp = skeleton.nonmissing_keypoints(with_torso=True)
                array = skeleton.keypoints_to_array(nonmissing_kp)
                x = array[:, 0]
                y = array[:, 1]
                ax.set_xlabel('')
                ax.set_ylabel('')
                # Connect keypoints to form skeleton
                for p1, p2 in connections:
                    if p1 in nonmissing_kp and p2 in nonmissing_kp:
                        start = skeleton.keypoints[p1]
                        end = skeleton.keypoints[p2]
                        ax.plot(np.linspace(start.x, end.x), np.linspace(start.y, end.y), c="blue", marker='|',
                                linestyle='-', linewidth=0.2)
                # Plot the dots
                ax.scatter(x, y, c='b', marker='o', linewidths=5.0)
                plt.title('Skeleton ' + str(skeleton.id) + "\nCluster: " + str(goal.actions[i]))
                plt.grid(True)
                # Puts text
                for label, keypoint in nonmissing_kp.items():
                        ax.text(keypoint.x, keypoint.y, label, None)
            fig.suptitle(str(goal.goal).upper())
            intention_id += 1
            previous_offset = offset
            # Reset
            skeleton_list = []
            last_id = 0
        plt.show()

    # Displays a global summary of the training process
    def summarize_training(self):
        self.training_result()  # Clusters and intentions
        self.show_clustering()      # Graphical visualization of the clusters
        self.plot_goal()            # Goals decompositions
