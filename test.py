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
from iCub import icub

from belief.bayesianNetwork import BeliefNetwork

# -------------------------------------------------------------------------------------------------------------------- #

# SIMULATION MODE

"""
from simulation.CognitiveArchitecture import CognitiveArchitecture as SimulatedCognitiveArchitecture

datapath = "/home/samuele/Research/datasets/block-building-game/"

cog = SimulatedCognitiveArchitecture()
cog.set_datapath(datapath)
cog.process(reload=False)
"""

#try:
    #time.sleep(1)
    #skeleton = icub.look_for_skeleton(icub.initialize_yarp_image(), 0)
    #skeleton.plot(dimensions=3)
#    game = BlockBuildingGame(debug=True)
#    game.collect_single_block("right")
#finally:
#    icub.cleanup()

#cog = CognitiveArchitecture(debug=True)
#cog.train(reload=True)
#print(cog.lowlevel.train.cluster_orientation_reach())
#goal = cog.read_intention()

#centroid = icub.observe_for_centroids(False)
#world_coordinates = icub.request_3d_points([list(centroid)])
#icub.take(world_coordinates[0])

#img = cv2.imread("/home/samuele/Research/datasets/block-building-game/test/castle-small/frame0001.jpg")
#img = cv2.imread("/home/samuele/Research/models/fullbody.jpg")
#skeleton = Skeleton(img, icub)
#skeleton.display(background=True, save=True, savename="fullbody", color=(0,255,255))
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


pass
