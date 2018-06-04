"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
from HighLevel import HighLevel
from TransitionQueue import TransitionQueue
from iCub import icub


class CognitiveArchitecture:
    def __init__(self, debug=False):
        self.tq = TransitionQueue()
        self.lowlevel = LowLevel(self.tq, debug)
        self.highlevel = HighLevel(self.tq)

    # Performs the training and learning
    def train(self, reload=False):
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        # Uses the training data to build the high-level model parameters
        self.highlevel.build_model(training_data)
        # Starts the high-level background thread to use it when needed
        self.highlevel.start()

    # Performs the intention reading (testing)
    def read_intention(self):
        # LowLevel decodes skeletons and tries to extract cluster transitions
        self.lowlevel.do_testing()
        # The above process ends when a goal has been inferred. Retrieve it
        current_goal = self.tq.get_goal_name()
        print("[DEBUG] " + self.__class__.__name__ + " reports goal: " + str(current_goal))
        return current_goal

    # Termination
    def terminate(self):
        self.highlevel.stop()
        icub.cleanup()
