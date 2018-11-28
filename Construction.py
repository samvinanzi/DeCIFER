"""

Class that represents a construction in the block building game.

"""

from enum import Enum


class Shape(Enum):
    SQUARE = 1
    HORIZONTAL_RECT = 2
    VERTICAL_RECT = 3


class Construction:
    def __init__(self, shape=None, red_blocks=0, blue_blocks=0):
        if shape is not None:
            assert isinstance(shape, Shape), "[ERROR] Construction: shape parameter must be of Shape type."
        self.shape = shape
        self.reds = red_blocks
        self.blues = blue_blocks

    def __eq__(self, other):
        return True if self.shape == other.shape and self.reds == other.reds and self.blues == other.blues else False

    def __str__(self):
        string = ""
        if self.shape == Shape.SQUARE:
            string += "Square "
        elif self.shape == Shape.HORIZONTAL_RECT:
            string += "Horizontal rectangle "
        else:
            string += "Vertical rectangle "
        string += "made up of "
        if self.reds > 0:
            string += str(self.reds) + " red " + ("blocks" if self.reds > 1 else "block")
            if self.blues > 0:
                string += " and "
        if self.blues > 0:
            string += str(self.blues) + " blue " + ("blocks" if self.blues > 1 else "block")
        return string
