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
        self.cognition = CognitiveArchitecture()
        self.coordinates = {
            "left": (-1.0, -0.5, -0.5),
            "right": (-1.0, 0.5, -0.5),
            "center": (0.0, 0.0, 0.0)
        }
        self.goals = {
            "tower": [],
            "wall": [],
            "castle": [],
            "stable": []
        }
        self.debug = debug
        icub.action_home()

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
        # Now the robot needs to learn the directions of movement
        cluster_orientations = self.cognition.lowlevel.train.cluster_orientation_reach()
        # Associate left / right movements for each goal
        for intention in self.cognition.lowlevel.train.intentions:
            for cluster in intention.actions:
                self.goals[intention.goal].append(cluster_orientations[cluster])
        return True

    # Reloads a previous training, to be able to play immediately
    def reload_training(self):
        icub.say("Let me remember the rules of the game...")
        self.cognition.train(reload=True)
        icub.say("Ok, done!")

    # Robot and human partner will play the game cooperatively
    def playing_phase(self):
        icub.say("Time to play! Feel free to start.")
        while True:
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            # If not unknown, perform an action
            #if goal == "clean":
            #    self.put_away()
            else:
                #self.collect_blocks(goal) ToDo re-enable
                pass
            # Asks the partner if to continue the game (only if task is not unknown)
            icub.say("Do you wish to continue?")
            if self.debug:
                response = icub.wait_and_listen_dummy()
            else:
                response = icub.wait_and_listen()
            if response != "yes":
                break

    # Determines where to obtain the blocks from
    def get_directions_sequence(self, transitions):
        # Filters out the center positions to obtain a sequence of only lefts and rights and deletes the first one which
        # is supposedly already been performed by the human
        return list(filter(lambda x: x != "center", transitions))[1:]

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
        #if object_coordinates[2] > 0:
        #    object_coordinates[2] *= -1
        while True:
            if icub.action_take(object_coordinates):
                icub.action_give()
                time.sleep(5)
                # while icub.is_holding():       # Busy waiting until the hand is freed
                #    time.sleep(1)
                break
            else:
                icub.say("Oh my, I wasn't able to grasp it. Let me try again.")
        icub.action_home()

    # Collects the blocks, in the order provided by the direction sequence
    def collect_blocks(self, goal):
        direction_sequence = self.get_directions_sequence(self.goals[goal])
        for direction in direction_sequence:
            self.collect_single_block(direction)
            time.sleep(2)

    # Receives a cube and puts it down in the toy chest
    def put_away(self):
        icub.action_expect()
        time.sleep(5)
        # while not icub.is_holding():       # Busy waiting until the hand is loaded
        #    time.sleep(1)
        icub.action_drop(self.coordinates['center'])
        icub.action_home()
