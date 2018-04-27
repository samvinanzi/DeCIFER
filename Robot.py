"""

This class models the iCub robot's capabilities.

"""

import yarp
from Listener import Listener
from queue import Queue
from asyncio import QueueEmpty
import numpy as np
import cv2
import pyttsx3 as tts
import time

# Initialise YARP
yarp.Network.init()

# Port names
LEFT_EYE = "/icub/camcalib/left/out"
RIGHT_EYE = "/icub/camcalib/right/out"
EYE_PORT = "/decifer/eye:i"
SFM_RPC_SERVER = "/SFM/rpc"
SFM_RPC_CLIENT = "/decifer/sfm/rpc"


class Robot:
    def __init__(self):
        # Initializations
        self.voice_commands = Queue()
        self.listener = Listener(self.voice_commands)
        self.accepted_commands = ["START", "STOP", "YES", "NO"]
        self.tts = tts.init()
        self.rpc_client = yarp.RpcClient()
        self.eye_port = yarp.Port()

        # Configurations
        self.tts.setProperty('rate', 120)
        self.tts.setProperty('voice', 'english')
        self.rpc_client.open(SFM_RPC_CLIENT)
        self.rpc_client.addOutput(SFM_RPC_SERVER)
        self.eye_port.open(EYE_PORT)
        self.eye_port.addOutput(LEFT_EYE)
        yarp.Network.connect(LEFT_EYE, EYE_PORT)

    # Text to Speech
    def say(self, phrase):
        self.tts.say(phrase)
        self.tts.runAndWait()

    # Analyses the vocal string in search of known commands or for a specific command
    def recognize_commands(self, response, listenFor=None):
        response = response.upper()
        if response in self.accepted_commands and (listenFor is None or response == listenFor):
            return response
        else:
            return None

    # Sends a request to the RPC server
    def request_3D_point(self, x, y):
        # Request Bottle
        req = yarp.Bottle()
        req.clear()
        # Response Bottle
        res = yarp.Bottle()
        res.clear()
        req.addString("Point")
        req.addDouble(x)
        req.addDouble(y)
        # RPC request
        self.rpc_client.write(req, res)
        return [float(i) for i in list(res.toString().split(' '))]

    # Streams the robot's vision
    def yarp_stream(self, fps=2):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((240, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
        i = 0
        while True:
            # Read the data from the port into the image
            self.eye_port.read(yarp_image)
            # Converts the color space to RGB
            frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # Saves the image
            cv2.imwrite("img/frames/" + str(i) + ".jpg", frame)
            # display the image that has been read
            #cv2.imshow('iCub eye', frame)
            #if cv2.waitKey(1) == 27:
            #    break  # esc to quit
            i += 1
            time.sleep(1/fps)
        # Cleanup
        #cv2.destroyAllWindows()

    # Closes all the open ports
    def cleanup(self):
        self.rpc_client.close()
        self.eye_port.close()

