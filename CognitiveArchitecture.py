"""

Initializes the low- and high-level modules and connects them to each other.

"""

from LowLevel import LowLevel
#from HighLevel import HighLevel
from BayesHighLevel import HighLevel
from TransitionQueue import TransitionQueue
from robots.robot_selector import robot
from Logger import Logger
import time
from Trust import Trust


class CognitiveArchitecture:
    def __init__(self, debug=False, offline=False, persist=False, enable_trust=False, simulation=False):
        self.tq = TransitionQueue()
        self.log = Logger()
        self.lowlevel = LowLevel(self.tq, self.log, debug, offline, persist)
        self.highlevel = HighLevel(self.tq)
        self.trust = Trust(self.log)
        self.trust_enabled = enable_trust
        self.simulation = simulation

    # Performs the training and learning
    def train(self, reload=False):
        # Initializes TRUST
        if self.trust_enabled:
            if reload:
                #self.trust.load_beliefs()   # Simply reloads... todo re-enable
                self.trust.initialize_trusted_trainer() # todo quick simulation initialization, change?
            elif not self.simulation:
                self.trust.learn_and_trust_trainer()    # Recognizes trainee's face and trustes him/her
            else:
                self.trust.initialize_trusted_trainer()     # No face data is captured
        # Initializes LOW-LEVEL
        training_data = self.lowlevel.reload_training() if reload else self.lowlevel.do_training()
        # Initializes HIGH-LEVEL
        # Strips the value '0' (neutral cluster pose)   # todo make it dynamic
        for entry in training_data:
            entry['data'] = [element for element in entry['data'] if element != 0]
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
    def read_intention(self, debug=False):
        # LowLevel decodes skeletons and tries to extract cluster transitions
        self.lowlevel.do_testing(self.simulation, debug)
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

    # Adds the outcome of an interaction in the logger
    def record_outcome(self, success):
        self.log.update_latest_success(success)
