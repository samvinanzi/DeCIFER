"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

"""

from CognitiveArchitecture import CognitiveArchitecture
from iCub import icub


class BlockBuildingGame:
    def __init__(self):
        self.cognition = CognitiveArchitecture()
        self.robot = icub
        self.goals = ["tower", "wall", "castle", "clean"]
        self.coordinates = {
            "Left": (0.0, 0.0),
            "Right": (0.0, 0.0),
            "Center": (0.0, 0.0)
        }

    # Trains the robot on the current rules of the game
    def training_phase(self):
        self.robot.say("Show me the rules of the game, I will learn them so that we can play together.")
        self.cognition.train()
        # Now the robot needs to learn the directions of movement

    # Robot and human partner will play the game cooperatively
    def playing_phase(self):
        while True:     # todo exit condition
            goal = self.cognition.read_intention()      # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "clean":
                icub.expect()
                icub.drop()
                icub.home()
