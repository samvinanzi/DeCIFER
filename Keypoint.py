"""

Keypoint class that represent coordinates for a 2D skeleton joint.

"""

import math


class Keypoint:
    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return Keypoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Keypoint(self.x - other.x, self.y - other.y)

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)   # Efficient distance between two points computation

    def median_with(self, other):
        x = (self.x + other.x) / 2
        y = (self.y + other.y) / 2
        return Keypoint(x, y)

    def is_empty(self):
        return True if self.x == 0.0 and self.y == 0.0 else False

    def as_list(self):
        return [self.x, self.y]

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ")"

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False


# Subclass of Keypoint that adds a third coordinate Z
class Keypoint3D(Keypoint):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        Keypoint.__init__(self, x, y)
        self.z = float(z)

    def __add__(self, other):
        return Keypoint3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Keypoint3D(self.x - other.x, self.y - other.y, self.z + other.z)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def median_with(self, other):
        x = (self.x + other.x) / 2
        y = (self.y + other.y) / 2
        z = (self.z + other.z) / 2
        return Keypoint3D(x, y, z)

    def is_empty(self):
        return True if self.x == 0.0 and self.y == 0.0 and self.z == 0.0 else False

    # Converts from the robot's ROOT to WORLD reference frames
    def root_to_world(self):
        temp = self.x
        self.x = self.y
        self.y = self.z
        self.z = -1 * temp

    def __str__(self):
        return "(" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ")"

    def __eq__(self, other):
        if self.x == other.x and self.y == other.y and self.z == self.z:
            return True
        else:
            return False
