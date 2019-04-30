from robots.AbstractRobot import AbstractRobot
from cv_bridge import CvBridge, CvBridgeError

import rospy
import intera_interface


class Sawyer(AbstractRobot):
    def __init__(self):
        super().__init__()

    def get_image_containers(self):
        pass

    def action_take(self, coordinates):
        pass

    def action_point(self, coordinates):
        pass

    def action_give(self):
        pass

    def action_expect(self):
        pass

    def action_home(self):
        pass

    def action_look(self, coordinates):
        pass

    def action_drop(self, coordinates):
        pass

    def look_for_skeleton(self, image_containers, i):
        pass

    def search_for_object(self):
        pass

    def evaluate_construction(self):
        pass

    def get_color(self):
        pass

    def cleanup(self):
        pass
