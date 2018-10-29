"""

Class that handles the logging needed to evaluate the results of the experiments.

"""

import datetime


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
               ("s;" if self.time_elapsed != 1 else ";")


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
        now = datetime.datetime.now()       # Gets the current date and time
        file_id = "{0}_{1}_{2}_{3}_{4}_{5}".format(str(now.year), str(now.month), str(now.day), str(now.hour),
                                                   str(now.minute), str(now.second))
        print("\n-----------COLLECTED DATA SUMMARY-----------")
        with open('logs/log_' + file_id + '.txt', 'w') as file:
            for i in range(0, len(self.trials)):
                line = "Trial " + str(i+1) + ": " + str(self.trials[i])
                print(line)             # Prints on console
                file.write(line + "\n")        # Saves on file
        print("--------------------------------------------")
