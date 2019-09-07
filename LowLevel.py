"""

This class represents the low-level model of the cognitive architecture. It performs training and execution of the
learning task.

"""

from Learner import Learner
from IntentionReader import IntentionReader


class LowLevel:
    def __init__(self, transition_queue, logger, debug=False, offline=False, persist=False):
        self.train = Learner(debug, persist=persist)
        self.test = IntentionReader(transition_queue, logger)
        self.offline = offline

    # Performs the training phase and outputs the training data
    def do_training(self):
        if self.offline:
            self.train.offline_learning()
        else:
            self.train.learn()
        return self.train.make_training_dataset()

    # Reloads previously computed training data
    def reload_training(self):
        path = "objects/"
        if self.offline:
            path += "offline/"
        self.train.reload_data(path)
        return self.train.make_training_dataset()

    # Updates the knowledge base
    def update_knowledge(self):
        self.train.update_knowledge()
        return self.train.make_training_dataset()

    # Performs skeleton extraction, clustering and transition analysis
    def do_testing(self, simulation=False):
        self.test.set_environment(self.train)
        if simulation:
            self.test.observer_simulator()
        else:
            if self.offline:
                self.test.offline_observe()
            else:
                self.test.observe()
