#!/usr/bin/env python

'''

This is a ROS node running on a ROS-enabled workstation which interacts directly with the Sawyer and listens to a socket
connection for incoming requests.

'''

import socket
import pickle
from messages import Request, Response
import cv2
from datetime import datetime
import time as t
from textwrap import wrap

from cv_bridge import CvBridge, CvBridgeError
import rospy
import intera_interface
from sensor_msgs.msg import Image
import message_filters

HOST = '10.0.0.90'
PORT = 65432
GRIPPER_CAM = "/io/internal_camera/right_hand_camera/image_rect"
HEAD_CAM = "/io/internal_camera/head_camera/image_rect"

class SawyerProxy:
	def __init__(self):
		self.close_request = False
		self.latest_frame = {
			'right_hand_camera': None,
			'head_camera': None
		}
		rospy.init_node('SawyerProxy')
		# Mechanical parts
		self.limb = intera_interface.Limb('right')
		self.head = intera_interface.Head()
		self.gripper = intera_interface.Gripper('right_gripper')
		self.cameras = intera_interface.Cameras()
		self.listen_and_respond()
	
	# Waits for a request
	def listen_and_respond(self):
		format_width = 55
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		s.settimeout(None)
		try:
			s.bind((HOST, PORT))
		except socket.error as msg:
			print(str(msg) + ". Bind failed.")
			quit(-1)
		print("Listening...\n")
		s.listen(5)
		# Accepts the connection
		conn, addr = s.accept()
		date, _ = self.get_datetime()
		print('+-' + '-' * format_width + '-+')
		print('| {0:^{1}} |'.format('Connected by ' + str(addr) + ' on ' + date, format_width))  
		while not self.close_request:
			try:
				start_time = t.time()
				# Reads the request
				data = conn.recv(4096)
				request = pickle.loads(data)
				assert isinstance(request, Request), "Received message was of an unsupported type."
				print('+-' + '-' * format_width + '-+')
				_, time = self.get_datetime()
				print('| {0:^{1}} |'.format('Time: ' + time, format_width))
				print('| {0:^{1}} |'.format(request, format_width))
				# Performs an operation
				response = self.route_request(request)      # Routing function
				# Sends back the data
				data = pickle.dumps(response)
				conn.send(data)
				elapsed = t.time() - start_time
				print('| {0:^{1}} |'.format(response, format_width))
				print('| {0:^{1}} |'.format("Elapsed: " + str(round(elapsed,2)) + "s", format_width))
				print('+-' + '-' * format_width + '-+')
			except KeyboardInterrupt:
				break
		# Closes the connection
		print('| {0:^{1}} |'.format('Closing connection', format_width))                
		print('+-' + '-' * format_width + '-+')
		conn.close()
		s.shutdown(socket.SHUT_RDWR)
		s.close()
		print('\nTerminated')
	
	# Fetches current date and time, in string format
	def get_datetime(self):
		ts = datetime.utcfromtimestamp(t.time())
		date = ts.strftime('%d/%m/%Y')
		time = ts.strftime('%H:%M:%S')
		return date, time

	# Manages the request, based on the command tag
	def route_request(self, request):
		if request.command == "TAKE":
			response = self.take()
		elif request.command == "POINT":
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
			response = self.camera()
		elif request.command == "CAMERA_GRIPPER":
			response = self.camera(gripper=True)
		elif request.command == "CLOSE":
			response = self.close()
		else:
			# Command not recognized
			response = Response(False, "Invalid command code")
		return response

	def take(self):
		#self.limb.move_to_joint_positions(cal_pos)
		return Response(True, None)

	def point(self):
		# todo move arm
		return Response(True, None)

	def give(self):
		# todo move arm
		return Response(True, None)

	def home(self):
		self.limb.move_to_neutral()
		return Response(True, None)

	def look(self, coordinates):
		#self.head.set_pan(head_angle, speed=0.3, timeout=0)
		return Response(True, None)

	def drop(self, coordinates):
		# todo move
		return Response(True, None)

	def camera(self, gripper=False):
		if gripper:
			camera_name = 'right_hand_camera'
		else:
			camera_name = 'head_camera'
		self.cameras.set_callback(camera_name, self.grab_single_frame, queue_size=1, callback_args=(camera_name))
		self.cameras.start_streaming(camera_name)
		# Busy waiting until one image is captured and stored in memory
		# First callback will deactivate streaming, hence signalling the availability
		while self.cameras.is_camera_streaming(camera_name):
			pass
		return Response(True, self.latest_frame[camera_name])
		
	# Camera callback. It captures one single frame before stopping the stream and stores it
	def grab_single_frame(self, img_data, camera_name):
		bridge = CvBridge()
		try:
			cv_image = bridge.imgmsg_to_cv2(img_data, "bgr8")
		except CvBridgeError, err:
			rospy.logerr(err)
			return
		self.latest_frame[camera_name] = cv_image
		self.cameras.stop_streaming(camera_name)
		cv2.imshow(camera_name, cv_image)
		cv2.waitKey(0)

	def close(self):
		self.close_request = True
		return Response(True, None)

# --------

def main():
	SawyerProxy()


if __name__ == "__main__":
	main()
