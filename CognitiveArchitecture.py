"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
from HighLevel import HighLevel
from TransitionQueue import TransitionQueue
from robots.robot_selector import robot
from Logger import Logger
import time


class CognitiveArchitecture:
    def __init__(self, debug=False, offline=False, persist=False):
        self.tq = TransitionQueue()
        self.log = Logger()
        self.lowlevel = LowLevel(self.tq, self.log, debug, offline, persist)
        self.highlevel = HighLevel(self.tq)

    # Performs the training and learning
    def train(self, reload=False):
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        # Uses the training data to build the high-level model parameters
        self.highlevel.build_model(training_data)
        # Starts the high-level background thread to use it when needed
        self.highlevel.start()

    # Updates a knowledge base
    # todo there might be a bug: can I stop and start the StopThread?
    def update(self):
        # Stops a running high-level
        self.highlevel.stop()
        # Performs the update
        training_data = self.lowlevel.update_knowledge()
        # Re-build the high level model and starts it
        self.highlevel.build_model(training_data)
        self.highlevel.start()

    # Performs the intention reading (testing)
    def read_intention(self):
        # LowLevel decodes skeletons and tries to extract cluster transitions
        self.lowlevel.do_testing()
        # The above process ends when a goal has been inferred. Retrieve it
        current_goal = self.tq.get_goal_name()
        self.tq.write_goal_name(None)   # Reset
        print("[DEBUG] " + self.__class__.__name__ + " reports goal: " + str(current_goal))
        return current_goal

    # Print the recorded data from the logger
    def print_log(self):
        self.log.print()

    # Termination
    def terminate(self):
        self.highlevel.stop()
        self.print_log()
        robot.cleanup()

    # DEBUG MODE -- Inserting manual observations into the transition queue for offline testing
    def debug_transition_input(self, observations):
        for item in observations:
            time.sleep(0.5)
            self.tq.put(item)
