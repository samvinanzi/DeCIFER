"""

Sandbox script

"""

import cv2
import time
from pathlib import Path
import os
from Learner import Learner
from Skeleton import Skeleton
from Keypoint import Keypoint
from IntentionReader import IntentionReader
from HighLevel import HighLevel
from CognitiveArchitecture import CognitiveArchitecture

# Workstation webcamera resolution
# wrk_camera_width = 800
# wrk_camera_height = 600


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


# Returns a formatted list of obsrevations from IntentionReading object
def build_observations(model):
    observations = []
    for intention in model.intentions:
        observations.append([intention.actions, len(intention.actions)])
    return observations


# -------------------------------------------------------------------------------------------------------------------- #

datapath = "/home/samuele/Research/datasets/block-building-game/"

cog = CognitiveArchitecture()
cog.set_datapath(datapath)
cog.process(reload=False)
