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
from LowLevel import LowLevel
from CognitiveArchitecture import CognitiveArchitecture
import yarp
import numpy as np
import matplotlib.pyplot as plt
import speech_recognition as sr
import pyaudio
from queue import Queue
from asyncio import QueueEmpty
import pyttsx3
from Robot import Robot
from Skeleton import NoHumansFoundException
from BlockBuildingGame import BlockBuildingGame
#from BlockBuildingGame2 import BlockBuildingGame
from iCub import icub
import pickle
from belief.bayesianNetwork import BeliefNetwork
from Logger import Logger

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


#cog = CognitiveArchitecture(debug=True)
#cog.train(reload=True)
#cog.lowlevel.train.do_pca()
#cog.lowlevel.train.clusters = []
#cog.lowlevel.train.intentions = []
#cog.lowlevel.train.generate_clusters()
#cog.lowlevel.train.generate_intentions()
#cog.lowlevel.train.show_clustering()
#for skeleton in cog.lowlevel.train.skeletons:
#    skeleton.check_skeleton_consistency()
#cog.lowlevel.train.plot_goal()
#cog.lowlevel.train.show_clustering(just_dots=False)
#cog.lowlevel.train.skeletons[10].plot(dimensions=3)
#cog.lowlevel.train.skeletons[30].plot(dimensions=3)
#cog.lowlevel.train.skeletons[50].plot(dimensions=3)
#cog.lowlevel.train.skeletons[70].plot(dimensions=3)


#centroid = icub.observe_for_centroids(False)
#world_coordinates = icub.request_3d_points([list(centroid)])
#icub.take(world_coordinates[0])

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

#game = BlockBuildingGame()
#icub.action_look(game.coordinates["left"])
#for i in range(0, 3):
#    print("Pass " + str(i))
#    n = icub.count_objects()
#    print(n)
#    icub.say(str(n))
#    time.sleep(3)


#icub.action_home()
#icub.action_look((-1.0, -0.5, -0.5))
#icub.say("3, 2, 1...")
#icub.say("Cheese!")
#skeleton = icub.look_for_skeleton(icub.initialize_yarp_image(), 0)
#skeleton.plot()
#skeleton.plot(dimensions=3)
#pickle.dump(skeleton, open("objects/test/skeleton/" + "initial.p", "wb"))


#skeleton_dict, skeleton_list = load_test_skeletons()
#for name, skeleton in skeleton_dict.items():
#    print(name)
#    skeleton.plot(dimensions=3)
#skeleton_dict["rightstop"].plot(dimensions=3)

#learner = Learner()
# I create manually skeletons, goals and offsets
#goal_list = ["bothwave", "rightstop", "initial", "leftstop", "leftplace", "rightplace"]
#offset_list = [0, 1, 2, 3, 4, 5]
# Set
#learner.skeletons = skeleton_list
#learner.goal_labels = goal_list
#learner.offsets = offset_list
# Do stuff
#learner.generate_dataset()
#learner.do_pca()
#learner.generate_clusters()
#learner.generate_intentions()
#learner.show_clustering()


#print("Starting...")
#sk = Skeleton(None, icub)
#sk.plot(dimensions=3)


#cog = CognitiveArchitecture(debug=True)
#cog.train(reload=False)
#cog.train(reload=True)
#cog.lowlevel.train.show_clustering()
#cog.read_intention()

bb = BlockBuildingGame(debug=True)
bb.reload_training()
bb.playing_phase()
bb.end()

'''
LOGGER TESTING
log = Logger()
log.new_trial()
log.update_latest_time()
log.update_latest_time()
time.sleep(2)
log.update_latest_time()
log.update_latest_goal("jumping")
log.new_trial()
time.sleep(1)
log.update_latest_time()
log.update_latest_goal("jack")
log.print()
'''

#icub.observe_for_centroid(display=True)

icub.cleanup()