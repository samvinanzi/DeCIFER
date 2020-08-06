"""

Simulates a batch of experiments. In particular, it simulates different informants with different behaviors.

"""

import random
from numpy.random import choice


class BatchSimulator:
    def __init__(self, n_trials=100):
        assert n_trials % 2 == 0, "The number of trials should be even"

        self.n_trials = n_trials
        self.goal_names = ['BGOR', 'BROG', 'GBRO', 'GORB', 'OGBR', 'ORBG', 'RBGO', 'ROGB']
        self.block_to_cluster = {
            'B': 3,
            'O': 4,
            'R': 6,
            'G': 5
        }
        # Tricker (untrustable) informant behavior (False = builds invalid sequence)
        self.informants = [[], [], [], []]
        self.informants[0] = [False] * self.n_trials
        self.informants[1] = [True] * (self.n_trials // 2) + [False] * (self.n_trials // 2)  # // forces int division
        self.informants[2] = [False] * (self.n_trials // 2) + [True] * (self.n_trials // 2)
        self.informants[3] = random.sample(self.informants[1], len(self.informants[1]))     # Random

        self.exposed_actions = []
        self.exposed_goals = []

        # Generates the simulated actions
        self.build_informant_behavior()

    # Picks a random goal
    def random_goal(self):
        return choice(self.goal_names, 1)[0]

    # Returns the first two cluster ids from the goal name
    def goal_to_cluster(self, goal):
        return [self.block_to_cluster[x] for x in list(goal)[:2]]

    # Based on the trustworthiness of the informant, it generates two cluster ids an a final construction sequence
    def generate_goal_and_action(self, correctness):
        goal = self.random_goal()
        id_list = self.goal_to_cluster(goal)
        if not correctness:
            return id_list, "BORG"  # BORG is an invalid block sequence
        else:
            return id_list, goal

    # Converts the informants design into actual actions
    def build_informant_behavior(self):
        for informant in self.informants:
            for i in range(len(informant)):
                informant[i] = self.generate_goal_and_action(informant[i])

    # Prepares the actions and goal lists to be consumed from the outside
    def expose_informant(self, id):
        assert id in range(4), "Exposed informant id must be 0-3"
        exposed = self.informants[id]
        for trial in exposed:
            self.exposed_actions.extend(trial[0])
            self.exposed_goals.append(trial[1])


sim = BatchSimulator(n_trials=10)
sim.expose_informant(3)
