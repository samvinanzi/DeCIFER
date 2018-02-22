"""

Sandbox script

"""

import PyOpenPose as OP
import cv2
import numpy as np
import time
import os
from Skeleton import Skeleton
import matplotlib.pyplot as plt
import simulations

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import pylab as pl

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pyclustering.cluster.xmeans as pyc
from pyclustering.utils import draw_clusters, read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES

from os import listdir
from os.path import isfile, join

import pickle



OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]
# Workstation webcamera resolution
wrk_camera_width = 800
wrk_camera_height = 600


# Shows the webcam stream
def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


# Retrieves a single camera image
def get_camera_image():
    cam = cv2.VideoCapture(0)  # 0 -> index of camera
    time.sleep(1)
    success, img = cam.read()
    if success:  # frame captured without any errors
        return img
    else:
        return None


def pca_test():
    iris = load_iris()
    pca = PCA(n_components=2).fit(iris.data)
    pca_2d = pca.transform(iris.data)
    pl.figure('Reference Plot')
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=iris.target)
    kmeans = KMeans(n_clusters=3, random_state=111)
    kmeans.fit(iris.data)
    pl.figure('K-means with 3 clusters')
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
    pl.show()


# Simulation of a data plot from multiple skeletons
def pca_sk_test():
    # Data preparation
    s1 = Skeleton(cv2.imread("img/test/2.jpg"))
    s2 = Skeleton(cv2.imread("img/test/3.jpg"))
    s3 = Skeleton(cv2.imread("img/test/4.jpg"))
    dataset = np.vstack((s1.as_feature(), s2.as_feature(), s3.as_feature()))
    # PCA computation
    pca = PCA(n_components=2).fit(dataset)
    pca_2d = pca.transform(dataset)
    pl.figure('Reference Plot')
    kmeans = KMeans(n_clusters=3, random_state=111)
    kmeans.fit(dataset)
    pl.figure('K-means with 3 clusters')
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
    pl.show()


# X-Means test
def xmeans_test():
    # Example data samples
    s1 = Skeleton(cv2.imread("img/test/2.jpg"))
    s2 = Skeleton(cv2.imread("img/test/3.jpg"))
    s3 = Skeleton(cv2.imread("img/test/4.jpg"))
    dataset = np.vstack((s1.as_feature(), s2.as_feature(), s3.as_feature()))
    # PCA to reduce dimensionality to 2D
    pca = PCA(n_components=2).fit(dataset)
    pca_2d = pca.transform(dataset).tolist()
    # create object of X-Means algorithm that uses CCORE for processing
    # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
    # let's avoid random initial centers and initialize them using K-Means++ method:
    initial_centers = kmeans_plusplus_initializer(list(pca_2d), 2).initialize()
    xmeans_instance = pyc.xmeans(pca_2d, initial_centers, ccore=True)
    # run cluster analysis
    xmeans_instance.process()
    # obtain results of clustering
    clusters = xmeans_instance.get_clusters()
    # display allocated clusters
    draw_clusters(pca_2d, clusters)


# Reads all images in a given folder and returns them as an array of images
def load_imageset(path):
    onlyfiles = sorted([f for f in listdir(path) if isfile(join(path, f))])
    images = np.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(path, onlyfiles[n]))
    return images


# Converts a video sequence in a skeleton sequence and saves it
def create_and_save():
    # Load images from desired folder
    images = load_imageset("/home/samuele/Research/datasets/CAD-60/data1/0512173548")    # Jar opening
    # Create skeletons for all of them
    skeletons = []
    for image in images:
        skeleton = Skeleton(image)
        skeletons.append(skeleton)
    # Saves the object
    pickle.dump(skeletons, open("objects/skeletons.p", "wb"))
    # Creates the dataset array
    dataset = np.zeros(shape=(1, 20))
    for skeleton in skeletons:
        # skeleton.display()
        dataset = np.vstack((dataset, skeleton.as_feature()))
    # Removes the first, empty row
    dataset = dataset[1:]
    # Saves the dataset for future uses
    pickle.dump(dataset, open("objects/dataset.p", "wb"))


# Performs X-Means clustering on the provided dataset
def clustering(dataset, dimensions=2, display=True):
    # Sanity check
    if dimensions != 2 and dimensions != 3:
        print("Clustering error: dimensions must be either 2 or 3.")
        return
    # PCA to reduce dimensionality to 2D
    pca = PCA(n_components=dimensions).fit(dataset)
    pca_2d = pca.transform(dataset).tolist()
    # create object of X-Means algorithm that uses CCORE for processing
    # initial centers - optional parameter, if it is None, then random centers will be used by the algorithm.
    # let's avoid random initial centers and initialize them using K-Means++ method:
    initial_centers = kmeans_plusplus_initializer(list(pca_2d), 2).initialize()
    xmeans_instance = pyc.xmeans(pca_2d, initial_centers, ccore=True)
    # run cluster analysis
    xmeans_instance.process()
    # obtain results of clustering
    clusters = xmeans_instance.get_clusters()
    if display:
        # display allocated clusters
        draw_clusters(pca_2d, clusters)
        # Find a way to print them nicely on a matplotlib


# --- TEST --- #

dataset = pickle.load(open("objects/dataset.p", "rb"))
clustering(dataset)