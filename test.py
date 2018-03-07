"""

Sandbox script

"""

import cv2
import time
from Learner import Learner
from Skeleton import Skeleton
from Keypoint import Keypoint
from IntentionReader import IntentionReader

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


# -------------------------------------------------------------------------------------------------------------------- #

# --- DATASET INITIALIZATION --- #

traindir = "/home/samuele/Research/datasets/block-building-game/train/"
testdir = "/home/samuele/Research/datasets/block-building-game/test/"
goal_names = ["tower", "wall", "castle-small", "clean"]

train = []
test = []
for goal in goal_names:
    train.append(traindir + goal)
    test.append(testdir + goal)

# --- PROCESSING --- #

env = Learner()
env.initialize(train)
#env.reload_data()
#env.plot_clusters()
#env.show_clustering()

ir = IntentionReader(env)
ir.initialize(test)

pass
