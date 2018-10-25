"""

This class represents the low-level model of the cognitive architecture. It performs training and execution of the
learning task.

"""

from Learner import Learner
from IntentionReader import IntentionReader


class LowLevel:
    def __init__(self, transition_queue, logger, debug):
        self.train = Learner(debug)
        self.test = IntentionReader(transition_queue, logger)

    # Performs the training phase and outputs the training data
    def do_training(self):
        self.train.learn()
        return self.train.make_training_dataset()

    # Reloads previously computed training data
    def reload_training(self):
        self.train.reload_data()
        return self.train.make_training_dataset()

    # Performs skeleton extraction, clustering and transition analysis
    def do_testing(self):
        self.test.set_environment(self.train)
        self.test.observe()
