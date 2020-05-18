"""

Hardware interface for the Sawyer robot.
Because of Python versions incompatibility, this class communicates via socket with a remote ROS node which interacts
directly with the robot.

"""

from robots.AbstractRobot import AbstractRobot
from messages import Request, Response, RemoteActionFailedException
from Skeleton import Skeleton, NoHumansFoundException
from BlockObserver import BlockObserver

import socket
import time
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle
pickle.HIGHEST_PROTOCOL = 2


class Sawyer(AbstractRobot):
    def __init__(self):
        super().__init__()
        self.HOST = '10.0.0.90'
        self.PORT = 65432
        self.socket = None
        # For Sawyer, coordinates are the joint angle configurationsvv
        self.coordinates = {        # These coordinates are used for looking. Sawyer doesn't need to move his head
            "left": "left",
            "right": "right",
            "center": "center",
        }
        self.block_coordinates = {      # Pickup locations for the blocks (name of the PoseLibrary pose)
            "BLUE": "blue",
            "ORANGE": "orange",
            "RED": "red",
            "GREEN": "green"
        }

    def connect_to_proxy(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0)
        try:
            value = self.socket.connect((self.HOST, self.PORT))
            return True
        except ConnectionError:
            print("[ERROR] Cannot contact remote SawyerProxy server! Is it up and running?")
            return False

    # Sends a network request to the ROS workstation and receives an answer back
    def send_proxy_request(self, request):
        camera_op = "CAMERA" in request.command     # Indicates if a camera image was requested
        data_out = pickle.dumps(request, protocol=2)    # Explicitly requests Python2 protocol
        while not self.connect_to_proxy():   # Open connection
            time.sleep(1)       # In case server is offline, continues to try
        self.socket.send(data_out)
        # Now wait for response
        try:
            if camera_op:            # If it's an image, incrementally receive all the data chunks
                data_in = b''
                while True:
                    block = self.socket.recv(4096)
                    if block:
                        data_in += block
                    else:
                        break
            else:
                data_in = self.socket.recv(4096)
            self.connection_close()   # Close the connection
            response = pickle.loads(data_in, encoding='latin1')     # To read a Python2 dump
            assert isinstance(response, Response), "Received message was of an unsupported type."
        except EOFError:
            response = Response(False, "Reception error")
        return response

    def connection_close(self):
        self.socket.shutdown(socket.SHUT_WR)
        self.socket.close()

    def cleanup(self):
        self.action_close()

    # NETWORK REQUESTS: sensing and actuation

    # Sends a request, collects the response, returns False in case of failure or the received data
    def request_action(self, command, parameters=None):
        request = Request(command, parameters)
        response = self.send_proxy_request(request)
        if response.status is False:
            print("[ERROR] " + str(response.values))
            raise RemoteActionFailedException
        else:
            return response.values

    # This function is here for retrocompatibility with the code written for iCub
    def get_image_containers(self):
        pass

    def get_camera_frame(self, gripper=False):
        if gripper:
            img = self.request_action("camera_gripper")
        else:
            img = self.request_action("camera_head")
        return np.asarray(img[0], dtype=np.uint8)

    def action_position_block(self, block, position):
        self.request_action("position", [block, position])

    def action_take(self, block):
        self.request_action("take", block)

    def action_point(self, block):
        self.request_action("point", block)

    def action_give(self):
        self.request_action("give")

    def action_expect(self):
        self.request_action("expect")

    def action_home(self):
        self.request_action("home")

    def action_look(self, coordinates):
        self.request_action("look", coordinates)    # todo define it in SawyerProxy

    def action_drop(self, coordinates):
        self.request_action("drop", coordinates)

    def action_midpose(self):
        self.request_action("midpose")

    def action_midpose_high(self):
        self.request_action("midpose_high")

    def action_ping(self):
        tic = time.time()
        self.request_action("ping")
        toc = time.time() - tic
        return toc

    def action_display(self, img_name):
        self.request_action("display", img_name)

    def action_say(self, text):
        self.request_action("say", text)

    def action_close(self):
        self.request_action("close")

    # Text to Speech
    def say(self, phrase):
        self.action_say(phrase)     # Displays the text on screen
        self.tts.say(phrase)
        print("[DEBUG] Robot says: " + phrase)
        self.tts.runAndWait()

    # This function does not make use of the image_containers parameter
    def look_for_skeleton(self, _, i):
        image = self.get_camera_frame()
        # Tries to extract the skeleton or raises a NoHumansFoundException
        skeleton = Skeleton(image, i)
        return skeleton

    # Makes the robot learn one goal
    # If debug = True, it only records a few samples and continues
    def record_goal(self, i=0, fps=2, debug=False):
        starting_i = i
        skeletons = []
        # Start the listener thread
        if not debug:
            stop_listening = self.recognizer.listen_in_background(self.microphone, self.speech_recognition_callback)
            print("[DEBUG] Listening in background")
        print("Robot is observing. Say \"STOP\" when the action is completed")
        while True:  # Begin the loop
            # Check for vocal commands
            if not debug and self.event.is_set():
                with self.lock:
                    response = self.vocal_queue.pop()
                self.event.clear()
                if self.recognize_commands(response, "STOP"):  # If the "stop" command was given, stop looping
                    break
            else:
                # If no vocal command was given, look for skeletons in camera image
                try:
                    skeleton = self.look_for_skeleton(None, i)  # Tries to extract the skeletal features
                    skeletons.append(skeleton)
                except NoHumansFoundException:
                    continue
                finally:
                    time.sleep(1 / fps)
                i += 1
                if debug and i - starting_i == 20:  # Debug mode, records 20 skeletons and continues (10s of action)
                    break
        # Stop the listener thread
        if not debug:
            stop_listening(wait_for_stop=True)
            print("[DEBUG] Listening stopped")
        # At this point, an action has just been performed and terminated
        # The robot looks at the final construction and gives it a name, based on the blocks disposition
        obs = BlockObserver()
        frame = self.get_camera_frame()
        obs.process(frame)
        goal_name = obs.label
        '''
        self.say("What goal did you just show me?")
        print("Waiting for the label...")
        # Waits until the goal name is given
        if debug:
            goal_name = self.wait_and_listen_dummy()
        else:
            goal_name = self.wait_and_listen()
        '''
        print("Set goal name to: " + goal_name)
        return skeletons, goal_name

    # Checks if the construction is a valid one
    def evaluate_construction(self):
        obs = BlockObserver()
        # Retrieves a camera image
        img = self.get_camera_frame()
        # Analyzes and validates it
        sequence, label, validity = obs.process(img)
        print("[DEBUG] Detected block sequence: " + str(label))
        return label, validity

    # Counts the blocks visible on the table
    def count_blocks(self):
        obs = BlockObserver()
        img = self.get_camera_frame()
        sequence, _, _ = obs.process(img)
        return len(sequence)

    def search_for_object(self):
        pass

    def get_color(self):
        pass
