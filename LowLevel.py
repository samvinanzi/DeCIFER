"""

This class represents the low-level model of the cognitive architecture. It performs skeletal extraction and clustering.

"""

from Learner import Learner
from IntentionReader import IntentionReader


class LowLevel:
    def __init__(self, robot, transition_queue):
        self.train = Learner(robot)
        self.test = IntentionReader(robot, transition_queue)
        self.transition_queue = transition_queue    # Cluster transition real-time queue

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
        self.test.start()

    # Stops the testing thread
    def stop_testing(self):
        self.test.stop_flag = True
        self.test.join(5.0)
