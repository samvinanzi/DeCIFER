"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

"""

from CognitiveArchitecture import CognitiveArchitecture
from iCub import icub
import time


class BlockBuildingGame:
    def __init__(self, debug=False):
        self.cognition = CognitiveArchitecture(debug)
        self.coordinates = {
            "left": (-1.0, -0.5, -0.5),
            "right": (-1.0, 0.5, -0.5),
            "center": (-2.0, 0.0, 0.25),
        }
        self.goals = {
            "tower": [],
            "wall": [],
            "castle": [],
            "stable": []
        }
        self.debug = debug
        icub.action_home()
        icub.action_look(self.coordinates["center"])

    # Main execution of the experiment
    def execute(self):
        if self.training_phase():
            self.playing_phase()
        else:
            icub.say("I'm sorry, something went wrong. Shall we try again?")
        self.end()

    # Terminates the experiment
    def end(self):
        icub.say("Thank you for playing!")
        self.cognition.terminate()      # Stops running threads, closes YARP ports, prints the recorded data log

    # The robot learns or remembers the directions of movement (i.e. what side the blocks are collected from)
    def set_orientations(self):
        # Now the robot needs to learn the directions of movement
        cluster_orientations = self.cognition.lowlevel.train.cluster_orientation_reach()
        # Associate left / right movements for each goal
        for intention in self.cognition.lowlevel.train.intentions:
            for cluster in intention.actions:
                self.goals[intention.goal].append(cluster_orientations[cluster])

    # Trains the robot on the current rules of the game
    def training_phase(self):
        icub.say("Show me the rules of the game, I will learn them so that we can play together.")
        self.cognition.train()
        # Checks that all the four goals have been learned
        for intention in self.cognition.lowlevel.train.intentions:
            if intention.goal not in self.goals:
                print("Goal " + intention.goal + " not included in the ones for this experiment! Please re-train.")
                return False
            else:
                print("Goal " + intention.goal + " correctly learned.")
        self.set_orientations()
        return True

    # Reloads a previous training, to be able to play immediately
    def reload_training(self):
        icub.say("Let me remember the rules of the game...")
        self.cognition.train(reload=True)
        self.set_orientations()
        if icub.AUTORESPONDER_ENABLED:
            icub.command_list = icub.command_list[0:-8]     # Truncates the automated response sequence
        icub.say("Ok, done!")

    # Robot and human partner will play the game cooperatively
    def playing_phase(self, point=False):
        turn_number = 1
        icub.say("Time to play! Feel free to start.")
        while True:
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            icub.say("We are building a " + goal)
            # If not unknown, perform an action
            self.collaborate(goal, point)
            # Asks the partner if to continue the game (only if task is not unknown)
            icub.say("Do you wish to continue with turn number " + str(turn_number+1) + "?")
            if self.debug:
                response = icub.wait_and_listen_dummy()
            else:
                response = icub.wait_and_listen()
            if response != "yes":
                break
            else:
                icub.say("Ok, go!")
                turn_number += 1

    # Determines where to obtain the blocks from
    def get_directions_sequence(self, transitions):
        # Filters out the center positions to obtain a sequence of only lefts and rights and deletes the first two which
        # have supposedly already been performed by the human
        return list(filter(lambda x: x != "center", transitions))

    # Perform collaborative behavior: collect of point the blocks
    def collaborate(self, goal, point=False):
        direction_sequence = self.get_directions_sequence(self.goals[goal])
        # For this experiment, consider only the last goal
        direction_sequence = [direction_sequence[-1]]
        for direction in direction_sequence:
            if point:
                self.point_single_block(direction)
            else:
                self.collect_single_block(direction)
            time.sleep(1)
        icub.action_look(self.coordinates["center"])    # Look back at the user

    # Looks to one side, seeks for a cube, picks it up and gives it to the human partner
    def collect_single_block(self, direction):
        icub.action_look(self.coordinates[direction])
        while True:
            object_centroid = icub.observe_for_centroid()
            if object_centroid is None:
                icub.say("I can't see any objects...")
                time.sleep(2)
            else:
                break
        object_coordinates = icub.get_object_coordinates(list(object_centroid))
        # Set manually the Z coordinate
        object_coordinates[2] = -0.02
        while True:
            if icub.action_take(object_coordinates):
                icub.action_give()
                time.sleep(5)
                while icub.is_holding():       # Busy waiting until the hand is freed
                    time.sleep(1)
                break
            else:
                icub.say("Sorry, I wasn't able to grasp it. Let me try again.")
        icub.action_home()

    def point_single_block(self, direction):
        point_coordinates = {
            "left": (-0.35, -0.25, -0.02),
            "right": (-0.35, 0.3, 0.03)
        }
        icub.action_look(self.coordinates[direction])
        icub.action_point(point_coordinates[direction])
        icub.action_home()
