"""

This class simulates the results from Experiment 2, specififcally the error rate of the real robot when observing the
experiment.

"""

from numpy.random import choice


class ObservationSampler:
    def __init__(self):
        # Blocks
        self.block_names = ['B', 'O', 'R', 'G']
        self.block_probabilities = {
            'B': [.93, .07, 0, 0],
            'O': [.18, .82, 0, 0],
            'R': [0, 0, .91, .09],
            'G': [0, 0, .06, .94],
        }
        # Goals
        self.goal_names = ['BGOR', 'BROG', 'GBRO', 'GORB', 'OGBR', 'ORBG', 'RBGO', 'ROGB']
        self.goal_probabilities = {
            'BGOR': [.8, .1, 0, 0, .1, 0, 0, 0],
            'BROG': [.25, .7, 0, 0, 0, .05, 0, 0],
            'GBRO': [0,	0, .9, .05, 0, 0, .05, 0],
            'GORB': [0,	0, .05, .9, 0, 0, 0, .05],
            'OGBR': [.2, 0, 0, 0, .75, .05, 0, 0],
            'ORBG': [0,	.15, 0,	0, 0, .85, 0, 0],
            'RBGO': [0,	0, .1, 0, 0, 0, .8, .1],
            'ROGB': [0, 0, 0, 0, 0, 0, .3, .7],
        }
        # Mapping
        self.cluster_to_block = {
            3: 'B',
            4: 'O',
            6: 'R',
            5: 'G'
        }

    # Samples at the block level (uses cluster ids as inputs and outputs)
    def sample_block(self, cluster_id):
        # Map the cluster id to block name
        block = self.cluster_to_block[cluster_id]
        try:
            weights = self.block_probabilities[block]
        except:
            print("No block " + str(block) + " in the probabilities table.")
            return False
        sample = choice(self.block_names, 1, p=weights)[0]
        # Map the block name to cluster id
        id = {v: k for k, v in self.cluster_to_block.items()}[sample]
        return id

    # Samples at the goal level
    def sample_goal(self, goal):
        try:
            weights = self.goal_probabilities[goal]
        except:
            print("No goal " + str(goal) + " in the probabilities table.")
            return False
        sample = choice(self.goal_names, 1, p=weights)[0]
        return sample
