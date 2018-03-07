"""

This class represents the high-level model of the cognitive architecture. It is modeled as a Hidden Markov Model.

"""

import numpy as np
from hmmlearn import hmm


class HighLevel:
    def __init__(self, states):
        self.state_names = states
        self.hmm = hmm.MultinomialHMM(n_components=len(states), startprob_prior=1.0, transmat_prior=1.0,
                                      algorithm='viterbi', random_state=None, n_iter=100, tol=0.01, verbose=False,
                                      params='ste', init_params='ste')

    # Parameter formatting
    # Input must be: [[obs1, len1], [obs2, len2], ... [obsN, lenN]]
    @staticmethod
    def produce_parameters(observations):
        X = np.array([])
        lengths = []
        for observation in observations:
            X = np.concatenate([X, observation[0]])
            lengths.append(observation[1])
        # Post-processing
        X = X.astype(int)
        X = X.reshape(-1, 1)
        return X, lengths

    # Train the model, based on sample observations
    # Input must be: [[obs1, len1], [obs2, len2], ... [obsN, lenN]]
    # (does it need iterations to avoid local minima?)
    def train_model(self, observations):
        X, lengths = HighLevel.produce_parameters(observations)
        self.hmm.fit(X, lengths)

    # Predict the probabilities of the states
    def predict(self, observations):
        X, lengths = HighLevel.produce_parameters(observations)
        return self.hmm.predict_proba(X, lengths)
