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
#goal_names = ["tower", "wall", "castle-small"]              # Reduced dataset, without "clean" goal

train = []
test = []
for goal in goal_names:
    train.append(traindir + goal)
    test.append(testdir + goal)

# --- PROCESSING --- #

# Training phase
env = Learner()
#env.initialize(train)      # New data
env.reload_data()           # Load old data
training_data = env.make_training_dataset()

# Then, build the high-level model on those actions
hl = HighLevel()
hl.build_model(training_data)

# Testing phase
ir = IntentionReader(env)
ir.initialize(test)
testing_data = ir.make_testing_dataset()

print(hl.predict(testing_data))
print(hl.decode(testing_data))

hl.incremental_decode(testing_data)

pass
