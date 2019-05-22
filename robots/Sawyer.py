"""

Hardware interface for the Sawyer robot.
Because of Python versions incompatibility, this class communicates via socket with a remote ROS node which interacts
directly with the robot.

"""

from robots.AbstractRobot import AbstractRobot
from messages import Request, Response

import socket
import pickle


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

    def get_image_containers(self):
        pass

    def action_take(self, coordinates):
        request = Request("take", coordinates)
        return self.send_proxy_request(request)

    def action_point(self, coordinates):
        print("[ERROR] Sawyer cannot point!")

    def action_give(self):
        pass

    def action_expect(self):
        pass

    def action_home(self):
        pass

    def action_look(self, coordinates):
        pass

    def action_drop(self, coordinates):
        pass

    def look_for_skeleton(self, image_containers, i):
        pass

    def search_for_object(self):
        pass

    def evaluate_construction(self):
        pass

    def get_color(self):
        pass

    def cleanup(self):
        pass
