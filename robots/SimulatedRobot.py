"""

Simulated robot, for local testing.

"""

from robots.AbstractRobot import AbstractRobot


class SimulatedRobot(AbstractRobot):
    def __init__(self, quiet=False):
        super().__init__()
        self.coordinates = {  # These coordinates are used for looking. Sawyer doesn't need to move his head
            "left": "left",
            "right": "right",
            "center": "center",
        }
        if not quiet:
            print("Warning: the simulated robot is operating in QUIET mode.")
        self.quiet = quiet

    def count_blocks(self):
        return 2

    # -- METHOD OVERRIDES --

    # Text to Speech
    def say(self, phrase):
        if not self.quiet:
            print("[SAY] " + phrase)

    # -- ABSTRACT METHODS --

    # Retrieve platform-dependent image containers
    def get_image_containers(self):
        print("[SIM_ROBOT] is accessing image containers")

    # Takes an object
    def action_take(self, coordinates):
        print("[SIM_ROBOT] is taking from " + str(coordinates))

    # Points to an object
    def action_point(self, coordinates):
        print("[SIM_ROBOT] is pointing at " + str(coordinates))

    # Gives an object
    def action_give(self):
        print("[SIM_ROBOT] is giving")

    # Requests an object
    def action_expect(self):
        print("[SIM_ROBOT] is expecting")

    # Returns in home position
    def action_home(self):
        print("[SIM_ROBOT] moves home")

    # Looks at one specific direction
    def action_look(self, coordinates):
        print("[SIM_ROBOT] looks " + str(coordinates))

    # Drops an object
    def action_drop(self, coordinates):
        print("[SIM_ROBOT] drops in " + str(coordinates))

    # Looks for a skeleton in a given image frame. Can raise NoHumansFoundException
    def look_for_skeleton(self, image_containers, i):
        print("[SIM_ROBOT] is looking for humans")

    # Searches for an object
    def search_for_object(self):
        print("[SIM_ROBOT] is searching for an object")

    # Evaluates the construction
    def evaluate_construction(self):
        print("[SIM_ROBOT] is evaluating the construction")

    # Determines the color of the object, provided the bounding box that encloses it
    def get_color(self):
        print("[SIM_ROBOT] is determining the color of the object")

    # Closes all the open ports
    def cleanup(self):
        print("[SIM_ROBOT] cleans up")
