"""

This class acts as a manager towards clusters and skeletons.

"""

from Skeleton import Skeleton
from Cluster import Cluster

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from sklearn.decomposition import PCA

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pyclustering.cluster.xmeans as pyc
from pyclustering.utils import draw_clusters

from os import listdir
from os.path import isfile, join

import pickle


class Controller:
    def __init__(self, reload=False, path="/home/samuele/Research/datasets/CAD-60/data1/0512173548"):
        self.skeletons = []
        self.dataset = []
        self.dataset2d = []
        self.clusters = []
        self.ax = None          # Plotting purpose
        # Data generation
        if not reload:
            self.generate_skeletons(path)
            self.generate_dataset()
            self.do_pca()
            self.generate_clusters()
            self.save()
        else:
            self.load()

    # Reads all images in a given folder and returns them as an array of images
    def load_imageset(self, path):
        onlyfiles = sorted([f for f in listdir(path) if isfile(join(path, f))])
        images = np.empty(len(onlyfiles), dtype=object)
        for n in range(0, len(onlyfiles)):
            images[n] = cv2.imread(join(path, onlyfiles[n]))
        return images

    # Converts a video sequence in a skeleton sequence and saves it
    def generate_skeletons(self, path):
        # Load images from desired folder
        images = self.load_imageset(path)  # Jar opening
        # Create skeletons for all of them
        id = 0
        for image in images:
            skeleton = Skeleton(image, id)
            self.skeletons.append(skeleton)
            id += 1

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self):
        # Creates the dataset array
        dataset = np.zeros(shape=(1, 20))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature()))
        # Removes the first, empty row
        self.dataset = dataset[1:]

    # Performs dimensionality reduction from 20-D to 2-D or 3-D through PCA
    def do_pca(self, dimensions=2):
        # Sanity check
        if dimensions != 2 and dimensions != 3:
            print("Clustering error: dimensions must be either 2 or 3.")
            return
        # PCA to reduce dimensionality to 2D
        pca = PCA(n_components=dimensions).fit(self.dataset)
        self.dataset2d = pca.transform(self.dataset).tolist()

    # Performs X-Means clustering on the provided dataset
    def generate_clusters(self):
        # create object of X-Means algorithm that uses CCORE for processing
        # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
        # let's avoid random initial centers and initialize them using K-Means++ method
        initial_centers = kmeans_plusplus_initializer(list(self.dataset2d), 2).initialize()
        xmeans_instance = pyc.xmeans(self.dataset2d, initial_centers, ccore=True,
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

    # Saves the objects in binary format
    def save(self):
        pickle.dump(self.skeletons, open("objects/skeletons.p", "wb"))
        pickle.dump(self.dataset, open("objects/dataset.p", "wb"))
        pickle.dump(self.dataset2d, open("objects/dataset2d.p", "wb"))
        pickle.dump(self.clusters, open("objects/clusters.p", "wb"))
        pickle.dump(self.ax, open("objects/ax.p", "wb"))

    # Loads the objects from binary format
    def load(self):
        self.skeletons = pickle.load(open("objects/skeletons.p", "rb"))
        self.dataset = pickle.load(open("objects/dataset.p", "rb"))
        self.dataset = pickle.load(open("objects/dataset.p", "rb"))
        self.dataset2d = pickle.load(open("objects/dataset2d.p", "rb"))
        self.clusters = pickle.load(open("objects/clusters.p", "rb"))
        self.ax = pickle.load(open("objects/ax.p", "rb"))

    # Displays a human-friendly result of the clustering operation
    def show_clustering(self):
        # Sanity check
        if self.ax is None:
            print("Error: must generate clusters before trying to display them.")
            raise RuntimeError
        # Create interactive plot
        for skeleton in self.skeletons:
            im = OffsetImage(skeleton.img, zoom=0.3)
            coordinates = self.dataset2d[skeleton.id]
            # Find the cluster id containing the skeleton (make it more pythonic)
            index = -1
            for i in range(len(self.clusters)):
                if skeleton.id in self.clusters[i].skeleton_ids:
                    index = i
                    break
            # Sanity check
            if index == -1:
                print("Error! Couldn't find cluster in which a skeleton belongs")
                raise RuntimeError
            color = self.clusters[index].color
            ab = AnnotationBbox(im, coordinates, bboxprops=dict(edgecolor=color))
            self.ax.add_artist(ab)
            # ax.text(coordinates[0]+0.0015, coordinates[1]+0.005, skeleton.id, fontsize=25)
        plt.show()
