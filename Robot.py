"""

Hardware interface for the iCub robot through YARP middleware.

"""

import yarp
import numpy as np
import cv2
import pyttsx3 as tts
import time
from Skeleton import Skeleton, NoHumansFoundException
import speech_recognition as sr
from threading import Lock, Event
import socket
import collections
from Construction import Shape, Construction

# Initialise YARP
yarp.Network.init()

# Port names
LEFT_EYE = "/icub/camcalib/left/out"        # Eye cameras output from the robot
RIGHT_EYE = "/icub/camcalib/right/out"
EYE_INPUT = "/decifer/eye:i"                # Eye camera input into DeCIFER
SFM_RPC_SERVER = "/SFM/rpc"
SFM_RPC_CLIENT = "/decifer/sfm/rpc"
ARE_CMD_RPC_SERVER = "/actionsRenderingEngine/cmd:io"
ARE_CMD_RPC_CLIENT = "/decifer/are/cmd/rpc"
ARE_GET_RPC_SERVER = "/actionsRenderingEngine/get:io"
ARE_GET_RPC_CLIENT = "/decifer/are/get/rpc"
LBP_BOXES = "/lbpExtract/blobs:o"
BOXES_INPUT = "/decifer/boxes:i"


class Robot:
    def __init__(self):
        self.remote_listener_ip = '127.0.0.1'    # ToDo set ! ! ! !
        self.remote_listener_port = 50106

        # Initializations
        self.vocal_queue = []
        self.accepted_commands = ["START", "STOP", "YES", "NO"]
        self.tts = tts.init()
        self.sfm_rpc_client = yarp.RpcClient()
        self.eye_port = yarp.Port()
        self.are_cmd_rpc_client = yarp.RpcClient()
        self.are_get_rpc_client = yarp.RpcClient()
        self.lbp_boxes_port = yarp.BufferedPortBottle()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()  # Default device

        # Synchronization
        self.lock = Lock()      # to synchronize access to vocal_queue
        self.event = Event()    # to signal the presence of new data in vocal_queue

        # Configurations
        self.tts.setProperty('rate', 140)                   # Text to Speech
        self.tts.setProperty('voice', 'english')
        self.sfm_rpc_client.open(SFM_RPC_CLIENT)            # SFM rpc
        self.sfm_rpc_client.addOutput(SFM_RPC_SERVER)
        self.eye_port.open(EYE_INPUT)                       # Eye input port and connection to external output port
        yarp.Network.connect(LEFT_EYE, EYE_INPUT)
        self.are_cmd_rpc_client.open(ARE_CMD_RPC_CLIENT)            # ARE 'cmd' rpc
        self.are_cmd_rpc_client.addOutput(ARE_CMD_RPC_SERVER)
        self.are_get_rpc_client.open(ARE_GET_RPC_CLIENT)            # ARE 'get' rpc
        self.are_get_rpc_client.addOutput(ARE_GET_RPC_SERVER)
        self.lbp_boxes_port.open(BOXES_INPUT)               # lbpExtract port and connection to external output port
        yarp.Network.connect(LBP_BOXES, BOXES_INPUT)

        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)  # we only need to calibrate once before we start listening

    # Text to Speech
    def say(self, phrase):
        self.tts.say(phrase)
        print("[DEBUG] Robot says: " + phrase)
        self.tts.runAndWait()

    # --- ACTION METHODS --- #

    # Defines a generic ActionsRenderingEngine rpc request
    # Returns True / False based on the success / failure of the action
    def are_request(self, command, *parameters):
        # Request Bottle
        req = yarp.Bottle()
        req.clear()
        # Response Bottle
        res = yarp.Bottle()
        res.clear()
        # Build request
        print("[DEBUG] ARE CMD REQUEST: \"" + str(command) + " " + " ".join([str(s) for s in list(parameters)]) + "\"")
        req.addString(command)
        for param in parameters:
            if isinstance(param, str):
                req.addString(param)
            elif isinstance(param, float):
                req.addDouble(param)
            elif isinstance(param, (list, tuple)):
                temp = req.addList()
                for element in param:
                    temp.addDouble(element)
            else:
                print("[ERROR] Unknown or invalid type for parameter: " + param)
                return False
        # RPC request
        self.are_cmd_rpc_client.write(req, res)
        time.sleep(2)  # Pause, to keep sequences of actions more ordered
        # Read response
        print("[DEBUG] ARE RESPONSE: " + res.toString())
        return res.get(0).asString().upper() == "ACK"

    # Takes an object
    def action_take(self, coordinates):
        return self.are_request("take", coordinates, "above")

    # Gives an object
    def action_give(self):
        return self.are_request("give")

    # Requests an object
    def action_expect(self):
        return self.are_request("expect")

    # Returns in home position
    def action_home(self):
        return self.are_request("home")

    # Looks at one specific direction
    def action_look(self, coordinates):
        return self.are_request("look", coordinates)

    # Drops an object
    def action_drop(self, coordinates):
        return self.are_request("drop", "over", coordinates)

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
        self.sfm_rpc_client.write(req, res)
        try:
            floatlist = [float(i) for i in list(res.toString().split(' '))]     # Converts the response from str to float
        except ValueError:
            floatlist = [0.0, 0.0, 0.0] * len(pointlist)      # In case of error, report a zero-value list
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
        assert image_containers is not None, "Missing image containers. Did you run initialize_yarp_image()?"
        img_array = image_containers[0]
        yarp_image = image_containers[1]
        # Gets the RGB frame from the left eye camera
        self.eye_port.read(yarp_image)
        # Converts the color space to RGB
        frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # Tries to extract the skeleton or raises a NoHumansFoundException
        skeleton = Skeleton(frame, i)
        return skeleton

    # Returns a tuple containing the centroid of one of the objects in the field of view. Optionally, displays it.
    def observe_for_centroid(self, display=False):
        time.sleep(1)       # Wait for vision to focus
        yarp.Network.connect(LBP_BOXES, BOXES_INPUT)
        #if yarp.Network.isConnected(LBP_BOXES, BOXES_INPUT):
        #    print("Connection ok!")
        trials = 0
        while True:
            time.sleep(1)
            btl = self.lbp_boxes_port.read(False)     # Fetches data from lbpExtract (True = blocking)
            if btl is None:
                print("[WARNING] No objects identified in the field of view")
                trials += 1
                if trials >= 10:
                    print("[WARNING] Object search failed")
                    return None
            else:
                bb_coords = [float(x) for x in btl.get(0).toString().split()]
                centroid = (int((bb_coords[0] + bb_coords[2]) / 2), int((bb_coords[1] + bb_coords[3]) / 2))
                print("[DEBUG] Centroid detected: " + str(centroid))
                if display:
                    img_array, yarp_image = self.initialize_yarp_image()
                    self.eye_port.read(yarp_image)
                    frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    cv2.circle(frame, (centroid[0], centroid[1]), 1, (255, 255, 255), 5)
                    cv2.imshow("Centroid location", frame)
                    cv2.waitKey(0)
                return centroid

    # Looks at the scene, counts red and non-red object and calculates the enclosing bounding box for the whole set.
    # Returns a dict with shape, red count and non-red count.
    def observe_for_shape_and_color(self, tolerance=0.5, percentage_threshold=70.0):
        # Step 1: fetching objects' bounding boxes
        time.sleep(1)  # Wait for vision to focus
        construction = Construction()
        bb_list = []
        bottle = self.lbp_boxes_port.read(False)  # Fetches data from lbpExtract (True = blocking)
        index = 0
        # There's probably a better way to do this that doesnt' involve an exception catching in an infinite loop to
        # detect the end of the set
        while True:
            try:
                bb_coords = [float(x) for x in bottle.get(index).toString().split()]    # [min_X, min_Y, max_X, max_Y]
            except AttributeError:
                print("[WARNING] No objects identified in the field of view")
                return None
            except IndexError:
                break
            print("[DEBUG] Object " + str(index) + " bounding box: " + str(bb_coords))
            bb_list.append(bb_coords)
            # Color inspection
            if self.is_object_red(bb_coords):
                construction.reds += 1
            else:
                construction.blues += 1
            index = index + 1
        # Step 2: calculating the total enclosing bounding box
        coords = np.array(bb_list)
        min_x = np.min(coords[:, 0])
        max_x = np.min(coords[:, 1])
        min_y = np.min(coords[:, 2])
        max_y = np.min(coords[:, 3])
        # Step 3: determination of the shape (square or long/tall rectangle)
        width = max_x - min_x
        height = max_y - min_y
        # Aspect ratio
        ar = width / height
        # A square will have an aspect ratio that is approximately equal to one, otherwise, the shape is a rectangle
        if (1 - tolerance) <= ar <= (1 + tolerance):
            construction.shape = Shape.SQUARE
        else:
            if width > height:
                construction.shape = Shape.HORIZONTAL_RECT
            else:
                construction.shape = Shape.VERTICAL_RECT
        return construction

    # Inspect an object's color and tests whever it is red.
    def is_object_red(self, boundingbox, percentage_threshold=70.0):
        # Retrieve and HSV image from the camera
        img_array, yarp_image = self.initialize_yarp_image()
        self.eye_port.read(yarp_image)
        frame = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        crop = frame[boundingbox[1]:boundingbox[3], boundingbox[0]:boundingbox[2]]  # Crop using the object's bounding box
        # Red has two HSV ranges, I need to match either.
        # lower mask (0-10)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(crop, lower_red, upper_red)
        # upper mask (170-180)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(crop, lower_red, upper_red)
        # join the masks
        mask = mask0 + mask1
        binary_mask = [int(bool(i)) for i in mask]      # Converts the mask to binary (1: red pixel, 0: non-red pixel)
        counter = collections.Counter(binary_mask)   # Counts occurencies of both elements
        tot = len(binary_mask)
        trues = counter[1]
        p = round(trues / tot * 100, 2)     # Percentage of truths
        return True if p > percentage_threshold else False

    # Requests the robot-centered coordinates of an object on the table using ARE "get s2c" rpc command
    def get_object_coordinates(self, centroid):
        # Request Bottle
        req = yarp.Bottle()
        req.clear()
        # Response Bottle
        res = yarp.Bottle()
        res.clear()
        # Build request
        print("[DEBUG] ARE GET REQUEST: get s2c " + str(centroid))
        req.addString("get")
        req.addString("s2c")
        temp = req.addList()
        for coordinate in centroid:
            temp.addDouble(coordinate)
        # RPC request
        self.are_get_rpc_client.write(req, res)
        # Read response
        print("[DEBUG] ARE GET RESPONSE: " + res.toString())
        stringlist = res.toString()
        return [float(i) for i in list(stringlist[1:-1].split(' '))]     # Converts the response from str to float

    # Counts the number of objects seen by the robot
    def count_objects(self):
        # Step 1: fetching objects' bounding boxes
        time.sleep(1)  # Wait for vision to focus
        bottle = self.lbp_boxes_port.read(False)  # Fetches data from lbpExtract (True = blocking)
        try:
            return int(bottle.size())
        except AttributeError:  # If a NoneType object is returned, that means that nothing was seen
            return 0

    # Returns True if the robot is holding an object, False otherwise
    def is_holding(self):
        # Request Bottle
        req = yarp.Bottle()
        req.clear()
        # Response Bottle
        res = yarp.Bottle()
        res.clear()
        # Build request
        req.addString("get")
        req.addString("holding")
        # RPC request
        self.are_get_rpc_client.write(req, res)
        # Read response
        print("[DEBUG] ARE GET RESPONSE: " + res.toString())
        return res.get(0).asInt()

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
        self.say("Ok")
        try:
            response = self.recognizer.recognize_google(audio, show_all=True)
            with self.lock:     # In case of exception, this lock won't be opened
                self.vocal_queue.append(response)
            self.event.set()
        except sr.UnknownValueError:
            print("[DEBUG] Google Speech Recognition could not understand audio")
            self.say("Sorry, I didn't understand. Can you please repeat?")
        except sr.RequestError as e:
            print("[DEBUG] Could not request results from Google Speech Recognition service; {0}".format(e))
            self.say("Sorry, I didn't understand. Can you please repeat?")

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

    # Remote microphone service (to use on iCub with no microphones onboard)
    def wait_and_listen_remote(self, ip=None):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if ip is not None:
                client_socket.connect((ip, self.remote_listener_port))
                print("Connected to " + ip + ":" + str(self.remote_listener_port))
            else:
                client_socket.connect((self.remote_listener_ip, self.remote_listener_port))
                print("Connected to " + self.remote_listener_ip + ":" + str(self.remote_listener_port))
            client_socket.send('listen'.encode('utf-8'))
            print("[DEBUG] Sent request, waiting for response...")
            response = client_socket.recv(1024).decode('utf-8')
        except socket.error as e:
            print("[ERROR] Connection error while connecting to " + self.remote_listener_ip)
            response = None
        client_socket.close()
        print("[DEBUG] Received: " + response)
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
                if debug and i - starting_i == 20:   # Debug mode, records 20 skeletons and continues (10s of action)
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
        self.sfm_rpc_client.close()
        self.are_cmd_rpc_client.close()
        self.eye_port.close()
