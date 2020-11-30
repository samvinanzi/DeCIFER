"""

Simulates a batch of experiments. In particular, it simulates different informants with different behaviors.

Informant success rates:

0 (H1) = 0%         |
1 (H2) = 50% + 0%   |>  deterministic
2 (H2) = 0% + 50%   |

3 (H4) = 50%        |
4 (H5) = 80%        |
5 (H6) = 20%        |
                    |>  randomized
6 (H7) = 90%        |
7 (H8) = 10%        |
8 (H9) = 60%        |

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
        self.n_informants = 9
        self.informants = [None] * self.n_informants
        self.informants[0] = [False] * self.n_trials
        self.informants[1] = [True] * (self.n_trials // 2) + [False] * (self.n_trials // 2)  # // forces int division
        self.informants[2] = [False] * (self.n_trials // 2) + [True] * (self.n_trials // 2)
        self.informants[3] = self.informants[1].copy()
        self.randomize(3)
        # "Natural" behaviors
        self.informants[4] = [True] * (self.n_trials * 80 // 100) + [False] * (self.n_trials * 20 // 100) # Expert
        self.randomize(4)
        self.informants[5] = [True] * (self.n_trials * 20 // 100) + [False] * (self.n_trials * 80 // 100) # Inexpert
        self.randomize(5)
        self.informants[6] = [True] * (self.n_trials * 90 // 100) + [False] * (self.n_trials * 10 // 100)  # Inexpert
        self.randomize(6)
        self.informants[7] = [True] * (self.n_trials * 10 // 100) + [False] * (self.n_trials * 90 // 100)  # Inexpert
        self.randomize(7)
        self.informants[8] = [True] * (self.n_trials * 60 // 100) + [False] * (self.n_trials * 40 // 100)  # Inexpert
        self.randomize(8)

        self.exposed_actions = []
        self.exposed_goals = []
        self.last_exposed_informant = None

        # Generates the simulated actions
        self.build_informant_behavior()

    # Randomizes an informant
    def randomize(self, id):
        assert 0 <= id <= self.n_informants, "Invalid id"
        assert self.informants[id] is not None, "Selected id has not been initialized"
        random.shuffle(self.informants[id])

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
        assert id in range(self.n_informants), "Exposed informant id must be 0-" + str(self.n_informants)
        exposed = self.informants[id]
        for trial in exposed:
            self.exposed_actions.extend(trial[0])
            self.exposed_goals.append(trial[1])
        self.last_exposed_informant = id
        print("Interacting with informant " + str(id))

    # Resets, after a trial. Please NOTE: this will not re-randomize the informants!
    def reset(self, randomize=False):
        # Security reset
        self.exposed_actions = []
        self.exposed_goals = []
        self.expose_informant(self.last_exposed_informant)


sim = BatchSimulator(n_trials=100)
#sim.expose_informant(3)
pass