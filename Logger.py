"""

Class that handles the logging needed to evaluate the results of the experiments.

"""


# Represents a single trial, that is one goal demonstration
class Trial:
    def __init__(self):
        self.goal_inferred = None
        self.time_elapsed = 0   # Number of transitions occoured before the prediction

    def update_time(self):
        self.time_elapsed += 1

    def update_goal(self, goalname):
        self.goal_inferred = goalname

    def __str__(self):
        return "Predicted \"" + self.goal_inferred + "\" after " + str(self.time_elapsed) + " observation" + \
               ("s." if self.time_elapsed != 1 else ".")


# Collector of trials, to evaluate the whole experiment
class Logger:
    def __init__(self):
        self.trials = []
        self.latest_index = None

    def new_trial(self):
        self.trials.append(Trial())
        self.latest_index = len(self.trials) - 1

    def update_latest_time(self):
        if self.latest_index is None:
            print("[ERROR] No trials to update")
        else:
            self.trials[self.latest_index].update_time()

    def update_latest_goal(self, goalname):
        if self.latest_index is None:
            print("[ERROR] No trials to update")
        else:
            self.trials[self.latest_index].update_goal(goalname)

    def print(self):
        for i in range(0, len(self.trials)):
            print("Trial " + str(i+1) + ": " + str(self.trials[i]))
