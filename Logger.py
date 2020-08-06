"""

Class that handles the logging needed to evaluate the results of the experiments.

"""

import datetime
import time


# Represents a single trial, that is one goal demonstration
class Trial:
    def __init__(self):
        self.goal_inferred = None
        self.time_elapsed = 0       # Number of transitions occoured before the prediction
        self.seconds = 0             # Time in seconds
        self.trust = None          # Trust value for the informant
        self.success = None

    def start(self):
        self.seconds = time.time()

    def update_time(self):
        self.time_elapsed += 1

    def update_goal(self, goalname):
        self.goal_inferred = goalname
        self.seconds = time.time() - self.seconds

    def update_trust(self, trust):
        self.trust = trust

    def update_success(self, success):
        self.success = success

    def __str__(self):
        return "Predicted \"" + self.goal_inferred + "\" after " + str(self.time_elapsed) + " observation" + \
               ("s" if self.time_elapsed != 1 else "") + " (" + str(self.seconds) + " seconds) [Trust: " + \
               str(self.trust) + "]... " + ("success!" if self.success else "failure")

    # Returns a CSV-writable representation of the trial
    def csv_repr(self):
        return self.goal_inferred + ";" + str(self.time_elapsed) + ";" + str(self.seconds) + ";" + str(self.trust) +\
               ";" + str(self.success) + ";"


# Collector of trials, to evaluate the whole experiment
class Logger:
    def __init__(self):
        self.trials = []
        self.latest_index = None

    def new_trial(self):
        self.trials.append(Trial())
        self.latest_index = len(self.trials) - 1
        self.trials[self.latest_index].start()

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

    def update_latest_trust(self, trust):
        if self.latest_index is None:
            print("[ERROR] No trials to update")
        else:
            self.trials[self.latest_index].update_trust(trust)

    def update_latest_success(self, success):
        if self.latest_index is None:
            print("[ERROR] No trials to update")
        else:
            self.trials[self.latest_index].update_success(success)

    def print(self):
        now = datetime.datetime.now()       # Gets the current date and time
        file_id = "{0}_{1}_{2}_{3}_{4}_{5}".format(str(now.year), str(now.month), str(now.day), str(now.hour),
                                                   str(now.minute), str(now.second))
        print("\n------------------------------COLLECTED DATA SUMMARY------------------------------")
        basename = 'logs/log_' + file_id
        filename_txt = basename + '.txt'
        filename_csv = basename + '.csv'
        with open(filename_txt, 'w') as file_txt, open(filename_csv, 'w') as file_csv:
            for i in range(0, len(self.trials)):
                line_txt = "Trial " + str(i+1) + ": " + str(self.trials[i])
                print(line_txt)                                             # Prints on console
                file_txt.write(line_txt + "\n")                             # Saves on TXT file
                line_csv = str(i+1) + ";" + self.trials[i].csv_repr()
                file_csv.write(line_csv + "\n")                             # Saves of CSV file
        print("----------------------------------------------------------------------------------")
        print("Saved TXT and CSV as: \"./" + basename + "\"")
