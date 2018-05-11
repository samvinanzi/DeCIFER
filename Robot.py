"""

Hardware interface for the iCub robot through YARP middleware.

"""

import yarp
from Listener import Listener
from queue import Queue
from asyncio import QueueEmpty
import numpy as np
import cv2
import pyttsx3 as tts
import time
from Skeleton import NoHumansFoundException

# Delete me
from Skeleton import Skeleton

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
        self.vocal_queue = Queue()
        self.listener = Listener(self.vocal_queue)
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
        yarp.Network.connect(LEFT_EYE, EYE_PORT)

    # Text to Speech
    def say(self, phrase):
        self.tts.say(phrase)
        self.tts.runAndWait()

    # Analyses the vocal string in search of known commands or for a specific command
    def recognize_commands(self, response, listenFor=None):
        if response is None:
            return None
        else:
            response = response.upper()
            if response in self.accepted_commands and (listenFor is None or response == listenFor):
                return response
            else:
                return None

    # Sends a request to the RPC server to convert a list of 2D points into 3D world coordinate wrt the robot
    # Input has the shape: [[x0, y0], ... , [xn, yn]]
    # Output has the shape: [[x0, y0, z0], ... , [xn, yn, zn]]
    def request_3d_points(self, pointlist):
        # Request Bottle
        req = yarp.Bottle()
        req.clear()
        # Response Bottle
        res = yarp.Bottle()
        res.clear()
        req.addString("Points")
        for couple in pointlist:
            req.addDouble(couple[0])
            req.addDouble(couple[1])
        # RPC request
        self.rpc_client.write(req, res)
        floatlist = [float(i) for i in list(res.toString().split(' '))]     # Converts the response from str to float
        splitlist = [floatlist[n:n + 3] for n in range(0, len(floatlist), 3)]   # Groups the values in X Y Z groups
        return splitlist

    # Busy waiting, listening for valid vocal input (commands are not processed at this stage)
    def wait_and_listen(self):
        while True:
            if not self.vocal_queue.empty():
                try:
                    response, status = self.vocal_queue.get_nowait()
                    self.vocal_queue.task_done()
                    if response is not None:
                        return str(response)
                except QueueEmpty:
                    print("[ERROR] Queue item is empty and cannot be read.")

    # Initializes the matrixes needed to fetch and process images
    def initialize_yarp_image(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((240, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
        return [img_array, yarp_image]

    # Looks for a skeleton in a given image frame. Can raise NoHumansFoundException
    def look_for_skeleton(self, image_containers):
        if not image_containers:
            print("[ERROR] yarp_image is not defined.")
        else:
            # Unpacks the matrixes
            img_array = image_containers[0]
            yarp_image = image_containers[1]
            # Gets the RGB frame from the left eye camera
            self.eye_port.read(yarp_image)
            # Converts the color space to RGB
            frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            # Tries to extract the skeleton or raises a NoHumansFoundException
            skeleton = Skeleton(frame, self)
            return skeleton

    # Makes the robot learn one goal
    def record_goal(self, fps=2):
        image_containers = self.initialize_yarp_image()
        i = 0
        skeletons = []
        # Start the Listener thread
        self.listener.start()
        print("Robot is observing. Say \"STOP\" when the action is completed")
        while True:     # Begin the loop
            # Check for vocal commands
            if not self.vocal_queue.empty():
                try:
                    response, status = self.vocal_queue.get_nowait()
                    self.vocal_queue.task_done()
                    if self.recognize_commands(response, "STOP"):
                        # If the "stop" command was given, stop looping
                        break
                except QueueEmpty:
                    print("[ERROR] Queue item is empty and cannot be read.")
            else:
                # If no vocal command was given, look for skeletons in camera image
                try:
                    skeleton = self.look_for_skeleton(image_containers)     # Tries to extract the skeletal features
                    skeleton.id = i
                    skeletons.append(skeleton)
                except NoHumansFoundException:
                    continue
                finally:
                    time.sleep(1 / fps)
                i += 1
        # At this point, an action has just been performed and terminated
        # Empy the vocal queue
        while not self.vocal_queue.empty():
            self.vocal_queue.get_nowait()
        self.say("What goal did you just show me?")
        print("Waiting for the label...")
        # Busy waiting until the goal name is given
        goal_name = self.wait_and_listen()
        print("Set goal name to: " + goal_name)
        # Stop the listener thread
        self.listener.stop_flag = True
        self.listener.join(5.0)
        return skeletons, goal_name

    # Closes all the open ports
    def cleanup(self):
        self.rpc_client.close()
        self.eye_port.close()
