"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
from HighLevel import HighLevel
from TransitionQueue import TransitionQueue


class CognitiveArchitecture:
    def __init__(self, debug=False):
        self.tq = TransitionQueue()
        self.lowlevel = LowLevel(self.tq, debug)
        self.highlevel = HighLevel(self.tq)

    # Performs the training and testing phases
    # todo exit condition
    def process(self, reload=False):
        # TRAINING
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        self.highlevel.build_model(training_data)

        # TESTING
        # Transitions are used in HighLevel to predict goals
        self.highlevel.start()
        # LowLevel decodes skeletons and tries to extract cluster transitions
        self.lowlevel.do_testing()
