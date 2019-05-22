"""

This is a ROS node running on a ROS-enabled workstation which interacts directly with the Sawyer and listens to a socket
connection for incoming requests.

"""

import socket
import pickle
from messages import Request, Response

#from cv_bridge import CvBridge, CvBridgeError
#import rospy
#import intera_interface

#rospy.init_node('Hello_Sawyer')
#limb = intera_interface.Limb('right')


class SawyerProxy:
    def __init__(self):
        self.HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
        self.PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

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
                response = self.route_request(request)
                # Sends back the data
                data = pickle.dumps(response)
                conn.send(data)
                print("[DEBUG] " + str(response))
                conn.close()
            except KeyboardInterrupt:
                print("[DEBUG] Closing.")
                s.close()

    # Analyzes the request and deals with it
    def route_request(self, request):
        if request.command == "TAKE":
            response = self.take()
        elif request.command == "GIVE":
            response = self.give()
        else:
            # Command not recognized
            response = Response(False, "Invalid command code")
        return response

    def take(self):
        # todo move arm
        return Response(True, None)

    def give(self):
        # todo move arm
        return Response(True, None)


# --------


def main():
    server = SawyerProxy()
    server.listen_and_respond()


if __name__ == "__main__":
    main()
