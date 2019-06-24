#!/usr/bin/env python

'''

Author:
	Samuele Vinanzi
Date:
	17/06/2019
Description:
	Use this script to obtain a dictionary representation of the robot's current pose, to be used to 
	animate the robot (for example, by using: intera_interface.Limb.move_to_joint_positions())
	
	`rosrun decifer posture_encoder.py`

'''

import rospy
import intera_interface
from intera_examples import JointRecorder
from intera_interface import CHECK_VERSION, Head
import os
import time
import csv

FILENAME = "joint_angles_tmp.txt"
CSV_PATH = "./src/decifer/share/csv/"

def posture_encoder():
	print("Initializing node... ")
	rospy.init_node("posture_encoder")
	rs = intera_interface.RobotEnable(CHECK_VERSION)
	print("Enabling robot... ")
	rs.enable()
	# Obtain the head pan
	pan = Head().pan()
	# Initialize the recorder
	recorder = JointRecorder(FILENAME, 100)
	print("> Press CTRL+C when in desired position <")
	recorder.record()
	# Open the file and read the last line
	with open(FILENAME) as pose_file:
		last_line = pose_file.readlines()[-1]
	# Split the commas
	values = [x.strip() for x in last_line.split(',')]
	# Delete first value (time)
	values = values[1:]
	# Encode the dictionary
	keys = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 
		'right_j5', 'right_j6', 'right_gripper']
	position = {}
	for key, value in zip(keys, values):
		position[key] = value
	# Delete the temp file
	os.remove(FILENAME)
	# Print the resulting dictionary
	print("\n\nposition = {")
	for key in sorted(position.keys()):
		print("\t'" + key + "': " + str(position[key]) + ",")
	print("}\n")
	print("head_pan = " + str(pan))
	# Save to CSV
	name = raw_input("\nName this posture: ")
	filename = CSV_PATH + name + '.csv'
	with open(filename, mode='w') as csv_file:
		    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		    csv_writer.writerow([position['right_gripper'], position['right_j0'], position['right_j1'], position['right_j2'], 
		    position['right_j3'], position['right_j4'], position['right_j5'], position['right_j6'], pan])
	print("Saved as: " + filename)
	print("\nTerminated.")

if __name__ == "__main__":
	posture_encoder()