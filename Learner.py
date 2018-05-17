"""

This class is in charge of learning intentions from training examples, managing everything that goes from skeleton
extraction to cluster generation.

"""

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


class Learner:
    def __init__(self, robot):
        self.skeletons = []  # Observed skeletons
        self.offsets = []  # Splits the dataset in sequences
        self.dataset = []  # 20-D dataset
        self.dataset2d = []     # 2-D dataset
        self.clusters = []      # Clusters
        self.intentions = []    # Intentions
        self.goal_labels = []   # Goal labels
        self.pca = None         # Trained parameters of a PCA model
        self.ax = None          # Plotting purpose
        self.robot = robot

    # --- INITIALIZATION METHODS --- #

    # Acquires and processes the training data
    def learn(self, savedir="objects/"):
        self.generate_skeletons()
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
        i=0
        while not finished:
            self.robot.say("Please, start demonstrating.")
            # Learns a single goal
            skeletons, goal_name = self.robot.record_goal(i)
            self.skeletons.extend(skeletons)    # extend instead of append to avoid nesting lists
            self.goal_labels.append(goal_name)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
            i += len(skeletons)
            self.robot.say("Do you want to show me another goal?")
            while True:
                response = self.robot.wait_and_listen()
                if self.robot.recognize_commands(response, listenFor="NO"):
                    self.robot.say("All right, thanks for showing me.")
                    finished = True
                    break
                elif self.robot.recognize_commands(response, listenFor="YES"):
                    self.robot.say("Ok then, let's continue.")
                    break
                else:
                    self.robot.say("Sorry, I didn't understand. Can you repeat?")

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self):
        # Creates the dataset array
        dataset = np.zeros(shape=(1, 30))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature()))
        # Removes the first, empty row
        self.dataset = dataset[1:]

    # Performs dimensionality reduction from 30-D to 2-D through PCA
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
            print("Error: more than 20 clusters detected, cannot display them all.")
            raise RuntimeError
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
            #goal_label = os.path.basename(self.goal_labels[offset_index])
            #intention.goal = goal_label
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
                im = OffsetImage(skeleton.img, zoom=0.08)
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
