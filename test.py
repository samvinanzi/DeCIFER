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
from Listener import Listener
from queue import Queue
from asyncio import QueueEmpty
import pyttsx3
from Robot import Robot


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


# Streams the robot's vision
def yarp_stream():
    # Initialise YARP
    yarp.Network.init()
    # Create a port and connect it to the iCub camera
    input_port = yarp.Port()
    input_port.open("/python-image-port")
    yarp.Network.connect("/icub/cam/right", "/python-image-port")
    # Create numpy array to receive the image and the YARP image wrapped around it
    img_array = np.zeros((240, 320, 3), dtype=np.uint8)
    yarp_image = yarp.ImageRgb()
    yarp_image.resize(320, 240)
    yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
    i = 0
    while True:
        # Read the data from the port into the image
        input_port.read(yarp_image)
        # Converts the color space to RGB
        frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Saves the image
        cv2.imwrite("img/frames/" + str(i) + ".jpg", frame)
        # display the image that has been read
        cv2.imshow('iCub eye', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
        i += 1
    # Cleanup
    input_port.close()
    cv2.destroyAllWindows()


# Microphone speech-to-text
def listen_to_speech():
    # create recognizer and mic instances
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    # adjust the recognizer sensitivity to ambient noise and record audio from the microphone
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Please, talk.")
        audio = recognizer.listen(source)
    try:
        response = recognizer.recognize_google(audio)
        status = "Ok"
    except sr.RequestError:
        # API was unreachable or unresponsive
        response = None
        status = "API unavailable"
    except sr.UnknownValueError:
        # speech was unintelligible
        response = None
        status = "Unable to recognize speech"
    return response, status


# Perform activities in background whilst waiting for thread signaling
def do_in_background():
    queue = Queue()
    l = Listener(queue)
    l.start()
    while True:
        if not queue.empty():
            try:
                response, status = queue.get_nowait()
                queue.task_done()
                print(response)
                print(status)
            except QueueEmpty:
                print("[ERROR] Queue item is empty and cannot be read.")
        else:
            print(". . .")
        time.sleep(1)

# -------------------------------------------------------------------------------------------------------------------- #

datapath = "/home/samuele/Research/datasets/block-building-game/"

#cog = CognitiveArchitecture()
#cog.set_datapath(datapath)
#cog.process(reload=False)

# YARP TESTING

# Initialise YARP
#yarp.Network.init()
#yarp_to_python()
#yarp_stream()

# SPEECH

#response, status = listen_to_speech()
#print("Response: " + response + "\nStatus: " + status)

#do_in_background()

robot = Robot()
robot.yarp_stream()
