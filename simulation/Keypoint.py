"""
Keypoint class that represents a 4-D vector containing coordinates and confidence value for a skeleton joint.
It can be used in simulation by just keeping constant the z-plane.
"""

import math


class Keypoint:
    def __init__(self, x=0.0, y=0.0, c=0.0):
        self.x = float(x)
        self.y = float(y)
        self.c = float(c)

    def __add__(self, other):
        avg_confidence = (self.c + other.c) / 2
        return Keypoint(self.x + other.x, self.y + other.y, avg_confidence)

    def __sub__(self, other):
        avg_confidence = (self.c + other.c) / 2
        return Keypoint(self.x - other.x, self.y - other.y, avg_confidence)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)   # Efficient distance between two points computation

    def median_with(self, other):
        x = (self.x + other.x) / 2
        y = (self.y + other.y) / 2
        c = (self.c + other.c) / 2
        return Keypoint(x, y, c)

    def is_empty(self):
        return True if self.x == 0.0 and self.y == 0.0 and self.c == 0.0 else False

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.c) + ")"
