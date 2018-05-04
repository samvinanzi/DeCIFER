"""

This class represents the low-level model of the cognitive architecture. It performs skeletal extraction and clustering.
TODO: online mode

"""

from simulation.Learner import Learner
from simulation.IntentionReader import IntentionReader
from pathlib import Path


class LowLevel:
    def __init__(self):
        self.train = Learner()
        self.test = IntentionReader()
        self.datasets = None

    # Sets the folder paths containing the data
    def set_datapaths(self, basedir):
        goal_names = [x.parts[-1] for x in Path(basedir + 'train/').iterdir() if x.is_dir()]    # Goals are the subdirs
        train_paths = []
        test_paths = []
        for goal in goal_names:
            train_paths.append(basedir + 'train/' + goal)
            test_paths.append(basedir + 'test/' + goal)
        self.datasets = {
            'train': train_paths,
            'test': test_paths
        }

    # Performs the training phase and outputs the training data
    def do_training(self):
        # Performs skeleton extraction, clustering and transition analysis
        self.train.initialize(self.datasets['train'])
        return self.train.make_training_dataset()

    # Reloads previously computed training data
    def reload_training(self):
        self.train.reload_data()
        return self.train.make_training_dataset()

    # Performs the testing (playing) phase and outputs the testing data
    def do_testing(self):
        self.test.set_environment(self.train)
        # Performs skeleton extraction, clustering and transition analysis
        self.test.initialize(self.datasets['test'])
        return self.test.make_testing_dataset()
