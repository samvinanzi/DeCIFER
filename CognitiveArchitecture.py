"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
from HighLevel import HighLevel
from queue import Queue
import keyboard     # temporary


class CognitiveArchitecture:
    def __init__(self, debug=False):
        self.transtion_queue = Queue()
        self.lowlevel = LowLevel(self.transtion_queue, debug)
        self.highlevel = HighLevel(self.transtion_queue)

    # Performs the training and testing phases
    def process(self, reload=False):
        # Training phase
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        self.highlevel.build_model(training_data)
        quit(-1) # todo delete this
        # Testing phase
        while True:
            # LowLevel decodes skeletons and tries to extract cluster transitions
            self.lowlevel.test.start()
            # Transitions are used in HighLevel to predict goals
            self.highlevel.start()
            # Temporary exit condition
            if keyboard.is_pressed('esc'):      # todo needs ROOT permissions to work => it will become a voice command
                # Stops the low-level thread
                self.lowlevel.test.stop()
                self.lowlevel.test.join(5.0)
                # Stops the high-level thread
                self.highlevel.stop()
                self.highlevel.join(5.0)
                break
