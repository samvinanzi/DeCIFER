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

#iris = load_iris()
#pca = PCA(n_components=2).fit(iris.data)
#pca_2d = pca.transform(iris.data)


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
    dataset = np.vstack((s1.get_example_record(), s2.get_example_record(), s3.get_example_record()))
    # PCA computation
    pca = PCA(n_components=2).fit(dataset)
    pca_2d = pca.transform(dataset)
    pl.figure('Reference Plot')
    kmeans = KMeans(n_clusters=3, random_state=111)
    kmeans.fit(dataset)
    pl.figure('K-means with 3 clusters')
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
    pl.show()


#image = cv2.imread("img/test/human.jpg")
#skeleton = Skeleton(image)

pca_sk_test()

#skeleton.plot()
#skeleton.show(background=True, save=False)