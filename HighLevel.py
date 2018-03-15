"""

This class represents the high-level model of the cognitive architecture. It is modeled as a Hidden Semi-Markov Model.

"""

import numpy as np
from hsmmlearn.hsmm import MultinomialHSMM


class HighLevel:
    def __init__(self):
        self.state_names = []
        self.hsmm = None

    # From an input training set, in dictionry form, computes the parameter matrixes and generates an HSMM
    def build_model(self, training_data):
        # Split data from input dictionary in separate lists
        data = []
        for entry in training_data:
            self.state_names.append(entry['label'])
            data.append(entry['data'])
        # Computate some utiliy variables
        n = len(self.state_names)               # Number of goals
        max_size = len(max(data, key=len))      # Max length of data sequences
        obs = list(set([item for sublist in data for item in sublist]))     # List of possible observations
        # HSMM parameter computation
        #   1) Transitions
        transitions = np.full((n, n), 1.0/n)    # Uniform probability distribution
        #   2) Durations
        durations = np.full((n, max_size), 0.0)
        for i in range(len(data)):
            durations[i][len(data[i])-1] = 1.0
        #   3) Emissions
        emissions = np.full((n, len(obs)), 0.0)
        i = 0
        for entry in data:
            # Counts the frequency of every observation in this training example
            for digit in obs:
                emissions[i][digit] = entry.count(digit) / len(entry)
            i += 1
        # HSMM model generation
        self.hsmm = MultinomialHSMM(emissions, durations, transitions, startprob=None, support_cutoff=100)

    # Decodes a sequence of observations to the most probable states that generated them
    def predict(self, observations):
        return self.hsmm.decode(observations)
