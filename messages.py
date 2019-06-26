#!/usr/bin/env python

"""

Socket messages exchanged with remote locations (e.g. Sawyer to/from SawyerProxy on a ROS machine).
They are explicitly declared as inheritants of Object for Python2 retrocompatibility.

"""

import numpy

# String, list
class Request(object):
	def __init__(self, command, parameters):
		assert isinstance(command, str), "Command parameter must contain a string value."
		self.command = command.upper()
		if isinstance(parameters, list) or parameters is None:
			self.parameters = parameters
		elif isinstance(parameters, str):
			self.parameters = parameters.encode("utf-8")
		else:
			self.parameters = [parameters]

	def __str__(self):
		message = "Request: " + self.command + " "
		if self.parameters:
			message += str(self.parameters)
		return message


# Bool, list
class Response(object):
	def __init__(self, status, values):
		assert isinstance(status, bool), "Status parameter must contain a string value."
		self.status = status
		if isinstance(values, list) or values is None:
			self.values = values
		else:
			self.values = [values]

	def __str__(self):
		message = "Response: " + str(self.status) + " "
		if self.values:
			if len(self.values[0]) > 30:
				message += "<image>"
			else:
				message += str(self.values)
		return message


# Custom exception
class RemoteActionFailedException(Exception):
	pass
