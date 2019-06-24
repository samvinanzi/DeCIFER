#!/usr/bin/env python

"""

Automatization of the postures for Sawyer robot. Use posture_encoder to generate CSV descriptors.

"""

import pickle
import csv
import os

class Posture:
	def __init__(self, name, configuration, head):
		self.name = name
		self.arm_angles = configuration
		self.head_angle = head
		
	def to_dict(self):
		return self.arm_angles, self.head_angle


class PostureLibrary:
	CSV_PATH = "./src/decifer/share/csv/"
	
	def __init__(self):
		self.postures = {}
		self.load_csv_library()
		
	# Reads the CSV files in the specified path to obtain Posture objects
	def load_csv_library(self):
		for file in os.listdir(PostureLibrary.CSV_PATH):
			if file.endswith(".csv"):
				path = PostureLibrary.CSV_PATH + file
				with open(path) as csv_file:
					file_name = os.path.splitext(os.path.basename(file))[0]
					csv_reader = csv.reader(csv_file, delimiter=',')
					for row in csv_reader:
						arm_angles = {
							'right_gripper': float(row[0]),
							'right_j0': float(row[1]),
							'right_j1': float(row[2]),
							'right_j2': float(row[3]),
							'right_j3': float(row[4]),
							'right_j4': float(row[5]),
							'right_j5': float(row[6]),
							'right_j6': float(row[7]),
						}
						head = float(row[8])
						new_posture = Posture(file_name, arm_angles, head)
						self.postures[file_name] = new_posture
						
	def get_posture(self, key):
		try:
			return self.postures[key].to_dict()
		except KeyError:
			raise PostureException
			
# Custom exception
class PostureException(Exception):
	pass
	
