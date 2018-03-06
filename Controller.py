"""

This class acts as a manager towards clusters and skeletons.

"""

from Skeleton import Skeleton
from Cluster import Cluster
from Intention import Intention

import cv2
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


class Controller:
    def __init__(self):
        self.skeletons = []
        self.dataset = []
        self.dataset2d = []
        self.clusters = []
        self.offsets = []       # Splits the dataset in sequences
        self.intentions = []
        self.ax = None          # Plotting purpose

    # Initialize a new Controller, generating data
    def initialize(self, path, savedir="objects/"):
        self.generate_skeletons(path)
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
        v = vars(self)
        for items in v:
            print(items)

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
        pickle.dump(self.skeletons, open(savedir + "skeletons.p", "wb"))
        pickle.dump(self.dataset, open(savedir + "dataset.p", "wb"))
        pickle.dump(self.dataset2d, open(savedir + "dataset2d.p", "wb"))
        pickle.dump(self.clusters, open(savedir + "clusters.p", "wb"))
        pickle.dump(self.offsets, open(savedir + "offsets.p", "wb"))
        pickle.dump(self.intentions, open(savedir + "intentions.p", "wb"))
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
        self.ax = pickle.load(open(path + "ax.p", "rb"))

    # Reads all images in a given folder and returns them as an array of images
    def read_imageset(self, path):
        onlyfiles = sorted([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        images = np.empty(len(onlyfiles), dtype=object)
        for n in range(0, len(onlyfiles)):
            images[n] = cv2.imread(os.path.join(path, onlyfiles[n]))
        return images

    # Converts a video sequence in a skeleton sequence and saves it
    def generate_skeletons(self, path):
        # If path is a single string and not a list, it converts it
        if not isinstance(path, list):
            path = [path]
        id = 0
        for folder in path:
            print("---Processing folder:" + folder)
            # Load images from desired folder
            images = self.read_imageset(folder)
            # Create skeletons for all of them
            for image in images:
                skeleton = Skeleton(image, id)
                self.skeletons.append(skeleton)
                id += 1
            self.offsets.append(id)

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self):
        # Creates the dataset array
        dataset = np.zeros(shape=(1, 20))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature()))
        # Removes the first, empty row
        self.dataset = dataset[1:]

    # Performs dimensionality reduction from 20-D to 2-D through PCA
    def do_pca(self):
        # PCA to reduce dimensionality to 2D
        pca = PCA(n_components=2).fit(self.dataset)
        self.dataset2d = pca.transform(self.dataset).tolist()

    # Performs X-Means clustering on the provided dataset
    def generate_clusters(self):
        # create object of X-Means algorithm that uses CCORE for processing
        # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
        # let's avoid random initial centers and initialize them using K-Means++ method
        initial_centers = kmeans_plusplus_initializer(list(self.dataset2d), 2).initialize()
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

    # Finds the closest centroid to a new skeletal sample
    def find_closest_centroid(self, sample):
        # Find the 2D coordinates of the new sample
        expanded_dataset = np.vstack((self.dataset, sample.as_feature()))
        pca = PCA(n_components=2).fit(expanded_dataset)
        expanded_dataset2d = pca.transform(expanded_dataset).tolist()
        sample2d = expanded_dataset2d[-1]
        # Compute the minimum distance to a centroid
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
        # Consider every sequence
        previous = 0
        for offset in self.offsets:
            intention = Intention()
            for i in range(previous, offset):
                # Retrieve the cluster id
                cluster_id = self.find_cluster_id(self.skeletons[i].id)
                if len(intention.actions) == 0 or intention.actions[-1] != cluster_id:
                    intention.actions.append(cluster_id)
            #intention.goal.append(...)     # todo goal definition
            self.intentions.append(intention)
            previous = offset

    # Plotss the clusters centroids
    def plot_clusters(self):
        x = []
        y = []
        for cluster in self.clusters:
            x.append(cluster.centroid[0])
            y.append(cluster.centroid[1])
            self.ax.text(cluster.centroid[0]+0.0015, cluster.centroid[1]+0.005, cluster.id, fontsize=25)
        plt.plot(x, y, 'kD')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cluster Centroids')
        plt.grid(True)
        plt.show()
