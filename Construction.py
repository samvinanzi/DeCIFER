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
        assert isinstance(shape, Shape), "[ERROR] Construction: shape parameter must be of Shape type."
        self.shape = shape
        self.reds = red_blocks
        self.blues = blue_blocks

    def __eq__(self, other):
        return True if self.shape == other.shape and self.reds == other.reds and self.blues == other.blues else False
