"""

Hardware interface for the iCub robot through YARP middleware.

"""

import yarp
import numpy as np
import cv2
import pyttsx3 as tts
import time
from Skeleton import NoHumansFoundException
import speech_recognition as sr
from threading import Lock, Event

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
        self.vocal_queue = []
        self.accepted_commands = ["START", "STOP", "YES", "NO"]
        self.tts = tts.init()
        self.rpc_client = yarp.RpcClient()
        self.eye_port = yarp.Port()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Default device

        # Synchronization
        self.lock = Lock()      # to synchronize access to vocal_queue
        self.event = Event()    # to signal the presence of new data in vocal_queue

        # Configurations
        self.tts.setProperty('rate', 140)
        self.tts.setProperty('voice', 'english')
        self.rpc_client.open(SFM_RPC_CLIENT)
        self.rpc_client.addOutput(SFM_RPC_SERVER)
        self.eye_port.open(EYE_PORT)
        yarp.Network.connect(LEFT_EYE, EYE_PORT)
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once before we start listening

    # Text to Speech
    def say(self, phrase):
        self.tts.say(phrase)
        print("[DEBUG] Robot says: " + phrase)
        self.tts.runAndWait()

    # --- VISION METHODS --- #

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

    # Initializes the matrixes needed to fetch and process images
    def initialize_yarp_image(self):
        # Create numpy array to receive the image and the YARP image wrapped around it
        img_array = np.zeros((240, 320, 3), dtype=np.uint8)
        yarp_image = yarp.ImageRgb()
        yarp_image.resize(320, 240)
        yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
        return [img_array, yarp_image]

    # Looks for a skeleton in a given image frame. Can raise NoHumansFoundException
    def look_for_skeleton(self, image_containers, i):
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
            skeleton = Skeleton(frame, self, i)
            return skeleton

    # --- SPEECH RECOGNITION METHODS --- #

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

    # This function is called by the background listener thread, if running, when a new audio signal is detected
    def speech_recognition_callback(self, recognizer, audio):
        print("[DEBUG] Detected audio. Recognizing...")
        try:
            response = self.recognizer.recognize_google(audio)
            with self.lock:     # In case of exception, this lock won't be opened
                self.vocal_queue.append(response)
            self.event.set()
        except sr.UnknownValueError:
            print("[DEBUG] Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("[DEBUG] Could not request results from Google Speech Recognition service; {0}".format(e))

    # Listens for valid vocal input (commands are not processed at this stage, but None responses are discarded as
    # they are founded in the queue)
    def wait_and_listen(self):
        # Starts the background recording
        stop_listening = self.recognizer.listen_in_background(self.microphone, self.speech_recognition_callback)
        print("[DEBUG] Listening in background")
        # non-busy-waiting for the listener to signal the production of new data
        self.event.wait()
        with self.lock:
            response = self.vocal_queue.pop()   # Secure consumption of the data
        self.event.clear()
        stop_listening(wait_for_stop=True)      # Makes sure that the background thread has stopped before continuing
        print("[DEBUG] Listening stopped")
        return response

    # Only for debugging purposes
    def wait_and_listen_dummy(self):
        response = input("Digit the word you would pronounce: ")
        return response

    # Makes the robot learn one goal
    # If debug = True, it only records a few samples and continues
    def record_goal(self, i=0, fps=2, debug=False):
        starting_i = i
        image_containers = self.initialize_yarp_image()
        skeletons = []
        # Start the listener thread
        if not debug:
            stop_listening = self.recognizer.listen_in_background(self.microphone, self.speech_recognition_callback)
            print("[DEBUG] Listening in background")
        print("Robot is observing. Say \"STOP\" when the action is completed")
        while True:     # Begin the loop
            # Check for vocal commands
            if not debug and self.event.is_set():
                with self.lock:
                    response = self.vocal_queue.pop()
                self.event.clear()
                if self.recognize_commands(response, "STOP"):   # If the "stop" command was given, stop looping
                    break
            else:
                # If no vocal command was given, look for skeletons in camera image
                try:
                    skeleton = self.look_for_skeleton(image_containers, i)     # Tries to extract the skeletal features
                    skeletons.append(skeleton)
                except NoHumansFoundException:
                    continue
                finally:
                    time.sleep(1 / fps)
                i += 1
                if debug and i - starting_i == 10:   # Debug mode, records 10 skeletons and continues
                    break
        # Stop the listener thread
        if not debug:
            stop_listening(wait_for_stop=True)
            print("[DEBUG] Listening stopped")
        # At this point, an action has just been performed and terminated
        self.say("What goal did you just show me?")
        print("Waiting for the label...")
        # Waits until the goal name is given
        if debug:
            goal_name = self.wait_and_listen_dummy()
        else:
            goal_name = self.wait_and_listen()
        print("Set goal name to: " + goal_name)
        return skeletons, goal_name

    # Closes all the open ports
    def cleanup(self):
        print("[DEBUG] Cleaning up...")
        self.rpc_client.close()
        self.eye_port.close()
