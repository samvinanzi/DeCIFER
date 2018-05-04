"""

Initializes the low- and high-level modules and connects them to each other.

"""

from simulation.LowLevel import LowLevel
from simulation.HighLevel import HighLevel


class CognitiveArchitecture:
    def __init__(self):
        self.lowlevel = LowLevel()
        self.highlevel = HighLevel()

    # Sets data folders
    def set_datapath(self, data_path):
        self.lowlevel.set_datapaths(data_path)

    # Performs the training and testing phases
    def process(self, reload=False):
        # Training phase
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        self.highlevel.build_model(training_data)
        # Testing phase
        testing_data = self.lowlevel.do_testing()
        self.highlevel.incremental_decode(testing_data)
