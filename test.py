"""

Sandbox script

"""

import cv2
import time
import os
import numpy as np
import pickle
from messages import Request, Response
from robots.robot_selector import robot
from BlockBuildingGame import BlockBuildingGame
from BlockObserver import BlockObserver


# -------------------------------------------------------------------------------------------------------------------- #

# SIMULATION MODE

"""
from simulation.CognitiveArchitecture import CognitiveArchitecture as SimulatedCognitiveArchitecture

datapath = "/home/samuele/Research/datasets/block-building-game/"

cog = SimulatedCognitiveArchitecture()
cog.set_datapath(datapath)
cog.process(reload=False)
"""

#robot = get_robot(iCub)

# Load the test skeletons
def load_test_skeletons():
    dict = {}
    lst = []
    path = "objects/test/skeleton/"
    files = [f for f in os.listdir(path) if f.endswith("p")]
    for file in files:
        filename, file_extension = os.path.splitext(file)
        skeleton = pickle.load(open(path + file, "rb"))
        dict[filename] = skeleton
        lst.append(skeleton)
    return dict, lst

def output():
    import yarp

    yp = yarp.Network()
    yp.init()

    port_in = yarp.BufferedPortBottle()
    port_in.open("/reader")
    if yp.connect("/lbpExtract/blobs:o", "/reader"):
        print("Connected")
    while True:
        time.sleep(1)
        btl = port_in.read(False)
        print(btl)
    #print("Disconnected")
    #port_in.close()


def cv2test(path="/home/samuele/blocks1_resized.jpg"):
    ## Read
    img = cv2.imread(path)
    img = cv2.GaussianBlur(img,(5,5),0)

    ## convert to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    ## mask of green (36,25,25) ~ (86, 255,255)
    # mask = cv2.inRange(hsv, (36, 25, 25), (86, 255,255))
    mask = cv2.inRange(hsv, (36, 25, 25), (85, 255, 255))
    #cv2.equalizeHist(mask, mask)

    ## slice the green
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(green, kernel, iterations=3)
    erosion = cv2.erode(dilation, kernel, iterations=3)


    edged = cv2.Canny(erosion, 150, 255)
    #edged = cv2.Sobel()

    cont = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## show
    cv2.imshow("im", cont)

    cv2.waitKey(0)


#img = cv2.imread("/home/samuele/Research/datasets/block-building-game/test/castle-small/frame0001.jpg")
#img = cv2.imread("/home/samuele/Research/models/fullbody.jpg")
#skeleton = Skeleton(img)
#skeleton.plot(save=False)
#skeleton.display(background=True, save=False, savename="fullbody", color=(0,255,255))
#skeleton.cippitelli_norm()
#skeleton.plot(dimensions=2)
#print("Orientation: " + skeleton.orientation_reach())

#print(icub.wait_and_listen_remote())

#bbn = BeliefNetwork("test", "belief/datasets/examples/tricker.csv")
#print(bbn.decision_making('A'))
#bbn_e = BeliefNetwork.create_episodic([bbn], 6)
#print(bbn_e.test_query(prettyTable=True))
#print(bbn_e.decision_making('A'))
#prediction = bbn.belief_estimation('A')
#rel = bbn.get_reliability()
#print("REL: " + str(rel))

"""
# Complete game
time.sleep(2)
bb = BlockBuildingGame(debug=True)
reload = True
if not reload:
    bb.training_phase()
else:
    bb.reload_training()
# Cluster editing for the video
#clusters = bb.cognition.lowlevel.train.clusters
#bb.cognition.lowlevel.train.clusters[1].id = 2
#bb.cognition.lowlevel.train.clusters[2].id = 1
#bb.cognition.lowlevel.train.show_clustering()
bb.playing_phase(point=True)
bb.end()
"""



#robot.action_home()
#robot.action_give()
#robot.say("Take this!")
#robot.action_home()
#robot.action_close()

#robot.action_home()
#robot.action_display("wall")
#time.sleep(1)
#robot.action_display("tower")
#time.sleep(1)
#robot.action_display("castle")
#time.sleep(1)
#robot.action_display("stable")
#robot.action_close()
#time.sleep(1)
#skeleton = robot.look_for_skeleton(None, 0)
#skeleton.display_fast()


#bb = BlockBuildingGame(debug=True)
#bb.reload_training()
#bb.playing_phase(point=True)
#bb.end()

#while True:
    #time.sleep(0.5)
    #image = robot.get_camera_frame()
    #cv2.imwrite("img/frames/" + str(i) + ".jpg", image)
    #cv2.imshow("Camera", image)
    #cv2.waitKey(500)

#time.sleep(3)
#image = robot.get_camera_frame()
#skeleton = robot.look_for_skeleton(None, 0)
#skeleton.display_fast()

'''
cobs = ColorObserver()
valid = cv2.imread("./img/valid.jpeg")
invalid = cv2.imread("./img/invalid.jpeg")
sequence = cobs.detect_cubes(valid)
print(cobs.validate_sequence())
sequence = cobs.detect_cubes(invalid)
print(cobs.validate_sequence())
#cobs.display()
'''

obs = BlockObserver()
img = cv2.imread("./img/blocks2/valid.jpeg")    # valid, invalid, incomplete, empty
sequence = obs.detect_sequence(img)
print(sequence)
print("Valid? " + str(obs.validate_sequence()))
print("Number of blocks: " + str(len(sequence)))
