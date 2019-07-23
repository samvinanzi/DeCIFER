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
#from BlockBuildingGame import BlockBuildingGame
from BlockBuildingGame2 import BlockBuildingGame
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

#bb = BlockBuildingGame(debug=True)
#bb.reload_training()
#bb.playing_phase(point=True)
#bb.end()

#time.sleep(3)
#image = robot.get_camera_frame()
#skeleton = robot.look_for_skeleton(None, 0)
#skeleton.display_fast()

'''
obs = BlockObserver()
img = cv2.imread("./img/blocks2/valid.jpeg")    # valid, invalid, incomplete, empty
sequence = obs.detect_sequence(img)
print(sequence)
print("Valid? " + str(obs.validate_sequence()))
print("Number of blocks: " + str(len(sequence)))
obs.display()
'''

'''
time.sleep(2)
robot.say("Go!")
for i in range(10):
    image = robot.get_camera_frame()
    cv2.imwrite("img/frames/" + str(i) + ".jpg", image)
    time.sleep(1)
robot.say("End")
'''

'''
bb = BlockBuildingGame(debug=True)
bb.training_phase()
bb.cognition.lowlevel.train.summarize_training()
'''

obs = BlockObserver()
img = cv2.imread("./img/blocks2/sawyer_valid.jpg")
print(obs.process(img))
#obs.display()
