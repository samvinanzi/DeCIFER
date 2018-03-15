"""

Sandbox script

"""

import cv2
import time
from Learner import Learner
from Skeleton import Skeleton
from Keypoint import Keypoint
from IntentionReader import IntentionReader
from HighLevel import HighLevel

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


data = [
         {
             'data': [0, 2, 0, 2, 0, 2, 0],
             'label': "tower"
         },
         {
             'data': [0, 1, 0, 1, 0, 1, 0],
             'label': "wall"
         },
         {
             'data': [0, 2, 0, 1, 0, 2, 0],
             'label': "castle"
         }
     ]
hl = HighLevel()
hl.build_model(data)
decoded_obs = hl.hsmm.decode([0, 2, 0, 2, 0, 2, 0, 0, 1, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 2, 0])

quit()





# First, show the low-level training actions
env = Learner()
#env.initialize(train)
env.reload_data()
#env.plot_clusters()
#env.show_clustering()

# Then, do the high-level training on those actions
#hl = HighLevel(env.goal_labels)
#training_observations = build_observations(env)
#hl.train_model(training_observations)

# Finally, observe testing actions and infer the goals
#ir = IntentionReader(env)
#ir.initialize(test)
#testing_observations = build_observations(ir)
#probs = hl.predict(testing_observations)

pass
