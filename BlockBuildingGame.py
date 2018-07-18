"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

"""

from CognitiveArchitecture import CognitiveArchitecture
from iCub import icub
import time


class BlockBuildingGame:
    def __init__(self):
        self.cognition = CognitiveArchitecture()
        self.coordinates = {        # todo calculate on the experimental setup
            "left": (0.0, 0.0, 0.0),
            "right": (0.0, 0.0, 0.0),
            "center": (0.0, 0.0, 0.0)
        }
        self.goals = {
            "tower": [],
            "wall": [],
            "castle": [],
            "clean": []
        }

    # Main execution of the experiment
    def execute(self):
        if self.training_phase():
            self.playing_phase()
        else:
            icub.say("I'm sorry, something went wrong. Shall we try again?")
        self.end()

    # Terminates the experiment
    def end(self):
        icub.cleanup()

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

    # Determines where to obtain the blocks from
    def get_directions_sequence(self, transitions):
        # Filters out the center positions to obtain a sequence of only lefts and rights and deletes the first one which
        # is supposedly already been performed by the human
        return list(filter(lambda x: x != "center", transitions))[1:]

    # Looks to one side, seeks for a cube, picks it up and gives it to the human partner
    def collect_single_block(self, direction):
        icub.look(self.coordinates[direction])
        object_centroid = icub.observe_for_centroids()
        world_coordinates = icub.request_3d_points([list(object_centroid)])
        icub.take(world_coordinates[0])
        icub.give()
        time.sleep(5)
        icub.home()

    # Collects the blocks, in the order provided by the direction sequence
    def collect_blocks(self, goal):
        direction_sequence = self.get_directions_sequence(self.goals[goal])
        for direction in direction_sequence:
            self.collect_single_block(direction)
            time.sleep(2)

    # Receives a cube and puts it down in the toy chest
    def put_away(self):
        icub.expect()
        time.sleep(5)
        icub.drop()
        icub.home()

    # Robot and human partner will play the game cooperatively
    def playing_phase(self):
        icub.say("Ok, time to play! Feel free to start.")
        while True:     # todo exit condition
            goal = self.cognition.read_intention()      # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":       # If unknown, just wait
                pass
            elif goal == "clean":       # Pick the cube
                self.put_away()
            else:
                self.collect_blocks(goal)
