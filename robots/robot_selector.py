"""

This script instantiates an object of the Robot class that will be shared through the rest of the architecture to
access the hardware controllers of a robot.

Currently implemented:
    - iCub
    - Sawyer

"""


from robots.Icub import iCub
from robots.Sawyer import Sawyer
from robots.SimulatedRobot import SimulatedRobot


def get_robot(name):
    robot_name = str(name).upper()
    if robot_name == 'ICUB':
        print("[DEBUG] Initializing an iCub robot...")
        return iCub()
    elif robot_name == 'SAWYER':
        print("[DEBUG] Initializing a Sawyer robot...")
        return Sawyer()
    elif robot_name == 'SIMULATED':
        print("[DEBUG] Initializing a Simulated robot...")
        return SimulatedRobot(quiet=True)
    else:
        print("[ERROR] get_robot: Invalid robot name input!")
        raise InvalidRobotException


# Custom exception
class InvalidRobotException(Exception):
    pass


# Change these lines to switch robot:

#robot = get_robot('iCub')
#robot = get_robot('Sawyer')
robot = get_robot('Simulated')
