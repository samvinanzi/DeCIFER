#!/usr/bin/env python

'''

This is a ROS node running on a ROS-enabled workstation which interacts directly with the Sawyer and listens to a socket
connection for incoming requests.

'''

import socket
from messages import Request, Response
import cv2
from datetime import datetime
import time as t
from textwrap import wrap
import os
from PIL import Image as ImagePIL
from PIL import ImageDraw, ImageFont
from cv_bridge import CvBridge, CvBridgeError
import rospy
import intera_interface
from sensor_msgs.msg import Image
import message_filters
from threading import Lock, Event
import numpy as np

try:
	import cPickle as pickle
except ImportError:
	import pickle
pickle.HIGHEST_PROTOCOL = 2

HOST = '10.0.0.90'
PORT = 65432
GRIPPER_CAM = "/io/internal_camera/right_hand_camera/image_rect"
HEAD_CAM = "/io/internal_camera/head_camera/image_rect"
IMG_PATH = "./src/decifer/share/images/"

frame_mutex = Lock()
request_event = Event()		# Frame Request Event
available_event = Event()		# Frame Available Event

class SawyerProxy:
	def __init__(self):
		self.in_home = False
		self.close_request = False
		self.latest_frame = None
		rospy.init_node('SawyerProxy')
		# Mechanical parts
		self.limb = intera_interface.Limb('right')
		self.head = intera_interface.Head()
		self.display = intera_interface.HeadDisplay()
		self.gripper = intera_interface.Gripper('right_gripper')
		self.cameras = intera_interface.Cameras()
		# Camera setup
		self.cameras.set_callback('right_hand_camera', self.grab_single_frame, queue_size=1000)
		self.cameras.set_callback('head_camera', self.grab_single_frame, queue_size=1000)
		self.cameras.start_streaming('head_camera')		# Head camera is the default streaming device
		# Work
		self.listen_and_respond()
	
	# Waits for a request, processes it and responds
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
		while not self.close_request:
			# Accepts the connection
			conn, addr = s.accept()
			_, time = self.get_datetime()
			print('+-' + '-' * format_width + '-+')
			#print('| {0:^{1}} |'.format('Connected by ' + str(addr) + ' on ' + time, format_width))  
			print('| {0:^{1}} |'.format('Connection from ' + str(addr[0]) + ":" + str(addr[1]) + 
				' at ' + time, format_width))  
			try:
				start_time = t.time()
				# Reads the request
				data = conn.recv(4096)
				request = pickle.loads(data)
				assert isinstance(request, Request), "Received message was of an unsupported type."
				print('+-' + '-' * format_width + '-+')
				#_, time = self.get_datetime()
				#print('| {0:^{1}} |'.format('Time: ' + time, format_width))
				print('| {0:^{1}} |'.format(request, format_width))
				camera_op = "CAMERA" in request.command
				# Performs an operation
				response = self.route_request(request)      # Routing function
				# Sends back the data
				data = pickle.dumps(response, protocol=2)
				if camera_op:
					conn.sendall(data)
				else:
					conn.send(data)
				#conn.shutdown(socket.SHUT_WR)
				elapsed = t.time() - start_time
				print('| {0:^{1}} |'.format(response, format_width))
				print('| {0:^{1}} |'.format("Time elapsed: " + str(round(elapsed,2)) + "s", format_width))
				print('+-' + '-' * format_width + '-+')
				# Closes the connection
				#print('| {0:^{1}} |'.format('Closing connection', format_width))                
				#print('+-' + '-' * format_width + '-+')
				conn.close()
			except KeyboardInterrupt:
				break
		# Terminate the server process
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
			response = self.take(request.parameters)
		elif request.command == "POINT":
			response = self.point(request.parameters)
		elif request.command == "GIVE":
			response = self.give()
		elif request.command == "MIDPOSE":
			response = self.mid_pose()
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
		elif request.command == "PING":
			response = self.ping()
		elif request.command == "DISPLAY":
			response = self.display_image(request.parameters)
		elif request.command == "SAY":
			response = self.say(request.parameters)
		elif request.command == "CLOSE":
			response = self.close()
		else:
			# Command not recognized
			response = Response(False, "Invalid command code")
		return response
		
	def camera(self, gripper=False):
		if gripper:
			camera_name = 'right_hand_camera'
		else:
			camera_name = 'head_camera'
		# If the current streaming device is not the desired one, it toggles it
		if not self.cameras.is_camera_streaming(camera_name):
			print "[M] Starting to stream"
			self.cameras.start_streaming(camera_name)	# All other streaming devices are forcefully closed
			t.sleep(1)		# Wait for camera to stabilize
		# Request a frame
		print "[M] Requesting a frame"
		request_event.set()
		print "[M] Waiting for a frame to be available"
		available_event.wait()
		print "[M] A frame seems available. Grabbing mutex"
		with frame_mutex:
			print "[M] Getting frame"
			frame = self.latest_frame
		print "[M] Out of mutex"
		print "[M] Clearing availability event"
		available_event.clear()
		return Response(True, frame)
		
	# Camera callback. It captures a frame from the camera stream and stores it
	def grab_single_frame(self, img_data):
		print "[C] I have a frame, do you need it?"
		if request_event.wait(0.0):
			print "[C] Yes, preparing to collect it"
			try:
				image = CvBridge().imgmsg_to_cv2(img_data, desired_encoding='bgr8')
			except CvBridgeError, err:
				rospy.logerr(err)
			# Resize to optimize transfer
			image = cv2.resize(image, (800, 600))
			print "[C] Grabbing mutex"
			with frame_mutex:
				print "[C] Saving frame"
				self.latest_frame = image
			print "[C] Out of mutex"
			print "[C] Notifing the availability of a frame"
			# Clear the request event
			request_event.clear()
			available_event.set()
			# Display the image
			self.display_temp_image(image)
		else:
			print "[C] No, dropping"
		print("[C] Exiting callback")
		
	# Displays some speech text on the display
	def say(self, message):
		# Prepares a white canvas
		canvas = ImagePIL.new('RGB', (1024, 600), color = (255, 255, 255))
		d = ImageDraw.Draw(canvas)
		# Sets the font
		fnt = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf', 100)
		# Prepares the string, breaking up lines when needed
		every=20
		message = '\n'.join(message[i:i+every] for i in xrange(0, len(message), every))
		# Writes
		d.text((10,10), message, font=fnt, fill=(0,0,0))
		# Saves it on disk in a temporary directory
		path = IMG_PATH + 'tmp_text.jpg'
		canvas.save(path)
		# Loads it on screen
		self.display.display_image(path)
		# Deletes the temp file
		os.remove(path)
		return Response(True, None)
		
	# Displays a white screen
	def whitescreen(self):
		canvas = ImagePIL.new('RGB', (1024, 600), color = (255, 255, 255))
		path = IMG_PATH + 'tmp_text.jpg'
		canvas.save(path)
		self.display.display_image(path)
		os.remove(path)
		
	# Go to a midway pose that avoids contact with the table
	def mid_pose(self):
		position = {
			'right_gripper': 0.041667,
			'right_j0': 0.602548828125,
			'right_j1': 0.645510742188,
			'right_j2': -2.72328027344,
			'right_j3': 1.92518847656,
			'right_j4': -0.163989257813,
			'right_j5': -0.914041015625,
			'right_j6': 4.45303320312,
		}
		self.limb.move_to_joint_positions(position)
		#return Response(True, None)

	def take(self, coordinates):
		self.mid_pose()
		#self.limb.move_to_joint_positions(cal_pos)
		self.in_home = False
		return Response(True, None)

	def point(self, coordinates):
		self.mid_pose()
		# todo move arm
		self.in_home = False
		return Response(True, None)

	def give(self):
		self.mid_pose()
		self.head.set_pan(-0.25)	# Looking forwards
		home_position = {
			'right_j0': 0.208775390625,
			'right_j1': 0.648694335938,
			'right_j2': -2.18177636719,
			'right_j3': 2.01313085937,
			'right_j4': 0.671734375,
			'right_j5': -1.30488183594,
			'right_j6': 3.97539257813,
			'right_gripper': 0.041667 
		}
		self.limb.move_to_joint_positions(home_position)
		self.in_home = False
		return Response(True, None)

	def home(self):
		self.whitescreen()
		if not self.in_home:
			self.mid_pose()
			self.head.set_pan(-1.41703617573)	# Looking forwards
			home_position = {
				'right_j0': 1.39565917969,
				'right_j1': 1.06604003906,
				'right_j2': -0.06905078125,
				'right_j3': -2.76589746094,
				'right_j4': -1.19899511719,
				'right_j5': 0.0740322265625,
				'right_j6': 3.17996289063,
				'right_gripper': 0.041667 
			}
			self.limb.move_to_joint_positions(home_position)
			self.in_home = True
		return Response(True, None)

	def look(self, coordinates):
		#self.head.set_pan(head_angle, speed=0.3, timeout=0)
		return Response(True, None)

	def drop(self, coordinates):
		self.mid_pose()
		# todo move
		self.in_home = False
		return Response(True, None)
			
	def ping(self):
		return Response(True, t.time())
		
	# Displays a temporary image on screen
	def display_temp_image(self, img):
		# Resize the image to display size 1024x600
		resized_img = cv2.resize(img, (1024, 600)) 
		# Saves the image on disk in a temporary directory
		path = IMG_PATH + 'tmp.jpg'
		cv2.imwrite(path, resized_img)
		# Loads it on screen
		self.display.display_image(path)
		# Deletes the temp file
		os.remove(path)
		
	# Displays an image from the library
	def display_image(self, img_name):
		name = img_name.decode('utf-8')
		path = IMG_PATH + name + '.jpg'
		if os.path.isfile(path):
			self.display.display_image(path)
			return Response(True, None)
		else:
			return Response(False, "Image '" + name + "' not found")

	def close(self):
		self.close_request = True
		return Response(True, None)

# --------

def main():
	SawyerProxy()


if __name__ == "__main__":
	main()
