"""
Keypoint class that represents a 4-D vector containing coordinates and confidence value for a skeleton joint.
It can be used in simulation by just keeping constant the z-plane.
"""

import math


class Keypoint:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __add__(self, other):
        return Keypoint(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Keypoint(self.x - other.x, self.y - other.y, self.z + other.z)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def median_with(self, other):
        x = (self.x + other.x) / 2
        y = (self.y + other.y) / 2
        z = (self.z + other.z) / 2
        return Keypoint(x, y, z)

    def is_empty(self):
        return True if self.x == 0.0 and self.y == 0.0 and self.z == 0.0 else False

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"
