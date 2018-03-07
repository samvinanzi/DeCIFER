"""

Base Class that contains the skeleton acquisition common methods

"""

from Skeleton import Skeleton
import numpy as np
import cv2
import os


class SkeletonAcquisitor:
    def __init__(self):
        self.skeletons = []     # Observed skeletons
        self.offsets = []       # Splits the dataset in sequences
        self.dataset = []       # 20-D dataset

    # Reads all images in a given folder and returns them as an array of images
    @staticmethod
    def read_imageset(path):
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
            images = SkeletonAcquisitor.read_imageset(folder)
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
