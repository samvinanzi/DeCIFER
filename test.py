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
"""
# Performs mean normalization on the keypoints  # TODO no! Must be done on all training examples
def mean_norm(self):
    # Remove from the computation the missing keypoints
    missing_indices = self.calculate_missing_indices()
    temp_keypoints = np.delete(self.keypoints, missing_indices, 0)

    minX = np.min(temp_keypoints[:, 0])
    minY = np.min(temp_keypoints[:, 1])

    maxX = np.max(temp_keypoints[:, 0])
    maxY = np.max(temp_keypoints[:, 1])

    meanX = np.mean(temp_keypoints[:, 0])
    meanY = np.mean(temp_keypoints[:, 1])

    # (x - mean) / (max - min)

    n_X = (temp_keypoints[:, 0] - meanX) / (maxX - minX)
    n_Y = (temp_keypoints[:, 1] - meanY) / (maxY - minY)

    n_X = n_X.reshape(-1,1)
    n_Y = n_Y.reshape(-1,1)

    self.keypoints = np.hstack([n_X, n_Y])
"""

image = cv2.imread("img/test/human.jpg")
skeleton = Skeleton(image)
skeleton.show(background=True)

#skeleton.show(background=True, save=False)
