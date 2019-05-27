"""

Hardware interface for the Sawyer robot.
Because of Python versions incompatibility, this class communicates via socket with a remote ROS node which interacts
directly with the robot.

"""

from robots.AbstractRobot import AbstractRobot
from messages import Request, Response, RemoteActionFailedException
from Skeleton import Skeleton, NoHumansFoundException

import socket
import pickle
import time


class Sawyer(AbstractRobot):
    def __init__(self):
        super().__init__()
        self.HOST = 'localhost'
        self.PORT = 65432

    # Sends a network request to the ROS workstation and receives an answer back
    def send_proxy_request(self, request):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect((self.HOST, self.PORT))
            except ConnectionError:
                print("[ERROR] Cannot contact remote SawyerProxy server! Is it up and running?")
                return None
            data_out = pickle.dumps(request, protocol=2)    # Explicitly requests Python2 protocol
            s.send(data_out)
            # Now wait for response
            try:
                data_in = s.recv(4096)
                response = pickle.loads(data_in, encoding='latin1')     # To read a Python2 dump
                assert isinstance(response, Response), "Received message was of an unsupported type."
            except EOFError:
                response = Response(False, "No response received")
        return response

    # NETWORK REQUESTS: sensing and actuation

    # Sends a request, collects the response, returns False in case of failure or the received data
    def request_action(self, command, parameters):
        request = Request(command, parameters)
        response = self.send_proxy_request(request)
        if response.status is False:
            print("[ERROR] " + response.values)
            raise RemoteActionFailedException
        else:
            return response.values

    # This function is here for retrocompatibility with the code written for iCub
    def get_image_containers(self):
        pass

    def get_camera_frame(self, hand=False):
        if hand:
            img = self.request_action("camera_hand", None)
        else:
            img = self.request_action("camera_head", None)
        return img

    def action_take(self, coordinates):
        self.request_action("take", coordinates)

    def action_point(self, coordinates):
        self.request_action("point", coordinates)

    def action_give(self):
        self.request_action("give", None)

    def action_expect(self):
        self.request_action("expect", None)

    def action_home(self):
        self.request_action("home", None)

    def action_look(self, coordinates):
        self.request_action("look", coordinates)

    def action_drop(self, coordinates):
        self.request_action("drop", coordinates)

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
        self.say("What goal did you just show me?")
        print("Waiting for the label...")
        # Waits until the goal name is given
        if debug:
            goal_name = self.wait_and_listen_dummy()
        else:
            goal_name = self.wait_and_listen()
        print("Set goal name to: " + goal_name)
        return skeletons, goal_name

    def search_for_object(self):
        pass

    def evaluate_construction(self):
        pass

    def get_color(self):
        pass

    def cleanup(self):
        pass
