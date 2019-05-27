"""

This is a ROS node running on a ROS-enabled workstation which interacts directly with the Sawyer and listens to a socket
connection for incoming requests.

"""

import socket
import pickle
from messages import Request, Response
import cv2

from cv_bridge import CvBridge, CvBridgeError
import rospy
import intera_interface
from sensor_msgs.msg import Image
import message_filters


class SawyerProxy:
    def __init__(self):
        self.HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        self.PORT = 65432  # Port to listen on (non-privileged ports are > 1023)
        rospy.init_node('SawyerProxy')
        self.limb = intera_interface.Limb('right')
        self.listen_and_respond()

    # Waits for a request
    def listen_and_respond(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((self.HOST, self.PORT))
        except socket.error as msg:
            print(str(msg) + ". Bind failed.")
            quit(-1)
        print("[DEBUG] Listening...")
        s.listen(5)
        while True:
            try:
                conn, addr = s.accept()
                print('Connected by ', addr)
                # Reads the request
                data = conn.recv(4096)
                request = pickle.loads(data)
                assert isinstance(request, Request), "Received message was of an unsupported type."
                print("[DEBUG] " + str(request))
                # Performs an operation
                response = self.route_request(request)      # Routing function
                # Sends back the data
                data = pickle.dumps(response)
                conn.send(data)
                print("[DEBUG] " + str(response))
                conn.close()
            except KeyboardInterrupt:
                print("[DEBUG] Closing.")
                s.close()

    # Manages the request, based on the command tag
    def route_request(self, request):
        if request.command == "TAKE":
            response = self.take()
        if request.command == "POINT":
            response = self.point()
        elif request.command == "GIVE":
            response = self.give()
        elif request.command == "HOME":
            response = self.home()
        elif request.command == "LOOK":
            response = self.look(request.parameters)
        elif request.command == "DROP":
            response = self.drop(request.parameters)
        elif request.command == "CAMERA_HEAD":
            response = self.camera(hand=False)
        elif request.command == "CAMERA_HAND":
            response = self.camera(hand=True)
        else:
            # Command not recognized
            response = Response(False, "Invalid command code")
        return response

    def take(self):
        # todo move arm
        return Response(True, None)

    def point(self):
        # todo move arm
        return Response(True, None)

    def give(self):
        # todo move arm
        return Response(True, None)

    def home(self):
        # todo move
        return Response(True, None)

    def look(self, coordinates):
        # todo move
        return Response(True, None)

    def drop(self, coordinates):
        # todo move
        return Response(True, None)

    def camera(self, hand):
        if hand:
            topic = "/internal_camera/right_hand_camera/image_rect"
        else:
            topic = "/internal_camera/head_camera/image_rect_color"
        msg = rospy.wait_for_message(topic, Image)
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        return Response(True, img)


# --------


def main():
    SawyerProxy()


if __name__ == "__main__":
    main()
