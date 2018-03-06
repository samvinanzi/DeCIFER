"""

Sandbox script

"""

import cv2
import time
from Controller import Controller
from Skeleton import Skeleton
from Keypoint import Keypoint

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


# ----- #


basedir = "/home/samuele/Research/datasets/frames2fps/"
directories = [
    basedir + "tower",
    basedir + "wall",
    basedir + "castle-small",
    basedir + "clean"
]

ctrl = Controller()
ctrl.initialize(directories)
ctrl.reload_data()
#ctrl.plot_clusters()
#ctrl.show_clustering()
pass
