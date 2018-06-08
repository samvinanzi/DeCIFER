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
from iCub import icub


# -------------------------------------------------------------------------------------------------------------------- #

# SIMULATION MODE

"""
from simulation.CognitiveArchitecture import CognitiveArchitecture as SimulatedCognitiveArchitecture

datapath = "/home/samuele/Research/datasets/block-building-game/"

cog = SimulatedCognitiveArchitecture()
cog.set_datapath(datapath)
cog.process(reload=False)
"""

#cog = CognitiveArchitecture(debug=True)
#cog.train(reload=True)
#goal = cog.read_intention()

icub.are_request("take", (0.1, 1.2, 0.8), "above")

pass