"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
from HighLevel import HighLevel
from Modes import Modes


class CognitiveArchitecture:
    def __init__(self, mode=Modes.OFFLINE):
        self.mode = mode
        self.lowlevel = LowLevel()
        self.highlevel = HighLevel()

    # Sets data folders
    def set_datapath(self, data_path):
        if self.mode == Modes.OFFLINE:
            self.lowlevel.set_datapaths(data_path)
        else:
            print("[ERROR] Cannot specify data paths in online mode.")
            quit(-1)

    # Performs the training and testing phases
    def process(self, reload=False):
        # Training phase
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        self.highlevel.build_model(training_data)
        # Testing phase
        testing_data = self.lowlevel.do_testing()
        self.highlevel.incremental_decode(testing_data)
