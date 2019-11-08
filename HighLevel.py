"""

This class represents the high-level model of the cognitive architecture. It is modeled as a Hidden Semi-Markov Model.

"""

import numpy as np
from hsmmlearn.hsmm import MultinomialHSMM
from StopThread import StopThread


class HighLevel(StopThread):
    def __init__(self, transition_queue):
        StopThread.__init__(self)
        self.state_names = []
        self.state_thresholds = []
        self.hsmm = None
        self.tq = transition_queue
        self.observations = []
        self.library = {}       # Used in the sequence matcher
        self.mapping = None

    # From an input training set, in dictionry form, computes the parameter matrixes and generates an HSMM
    # Ratio is the percetage that observed duration should be the taught one
    def build_model(self, training_data, ratio=0.9, threshold_percentage=50):
        # Sanity check
        if ratio <= 0.0 or ratio > 1.0:
            print("[Error] Invalid ratio value.")
            quit(-1)
        # Split data from input dictionary in separate lists
        data = []
        for entry in training_data:
            #entry['data'] = entry['data'][:-2]
            self.state_names.append(entry['label'])
            data.append(entry['data'])
            # Thresholds are computed as rounded <var>% of duration of that state, minimum 1
            self.state_thresholds.append(max(1, int(round(len(entry['data']) / 100.0 * threshold_percentage))))
            # Library storage
            self.library[entry['label']] = entry['data']
        # Compute some utiliy variables
        n = len(self.state_names)               # Number of goals
        max_size = len(max(data, key=len))      # Max length of data sequences
        obs = list(set([item for sublist in data for item in sublist]))     # List of possible observations
        self.mapping = dict(zip(obs, np.arange(len(obs))))  # Maps the L-level symbols to a [0,n] range H-level symbols
        # HSMM parameter computation
        #   1) Transitions
        transitions = np.full((n, n), 1.0/n)    # Uniform probability distribution
        #   2) Durations
        durations = np.full((n, max_size), (1.0-ratio)/(max_size-1))    # (1-ratio)% chance to have a different duration
        for i in range(len(data)):
            durations[i][len(data[i])-1] = ratio              # (ratio)% chace to have normal duration
        #   3) Emissions
        emissions = np.full((n, len(obs)), 0.0)
        i = 0
        for entry in data:
            # Counts the frequency of every observation in this training example
            for digit in obs:
                emissions[i][self.observation_to_map(digit)] = entry.count(digit) / len(entry)
            i += 1
        # HSMM model generation
        self.hsmm = MultinomialHSMM(emissions, durations, transitions, startprob=None, support_cutoff=100)

    def observation_to_map(self, observation):
        if not isinstance(observation, list):
            observation = [observation]
        try:
            map_list = []
            for x in observation:
                map_list.append(self.mapping[x])
            return map_list
        except KeyError:
            return None

    def map_to_observation(self, map):
        return next((key for key, value in self.mapping.items() if value == map), None)

    # Infers a sequence of observations to the most probable states that generated them
    def predict(self, observations):
        mapped_observations = self.observation_to_map(observations)
        return self.hsmm.decode(mapped_observations)

    # Generates a sequence of goal labels that correspond to the predictions
    def decode(self, observations):
        states = self.predict(observations)
        return [self.state_names[i] for i in states]

    # Decodes observations incrementally
    def incremental_decode_batch(self):
        for i in range(1, len(self.observations)):
            states = self.decode(self.observations[0:(i+1)])
            print(self.observations[0:(i+1)])
            print(states)
            states.reverse()
            item = states[0]
            count = 0
            for state in states:
                if state == item:
                    count += 1
                else:
                    break
            if count >= self.state_thresholds[self.state_names.index(item)]:
                current_goal = item
            else:
                current_goal = None
            print("Current inferred goal is: " + (current_goal if current_goal is not None else "unknown") + "\n")

    # Decodes observations incrementally
    def incremental_decode(self):
        if len(self.observations) > 1:
            states = self.decode(self.observations)
            print(self.observations)
            print(states)
            states.reverse()
            item = states[0]
            count = 0
            for state in states:
                if state == item:
                    count += 1
                else:
                    break
            if count >= self.state_thresholds[self.state_names.index(item)]:
                current_goal = item
            else:
                current_goal = None
            print("Current inferred goal is: " + (current_goal if current_goal is not None else "unknown") + "\n")
            return current_goal

    # Accesses the transition queue and decodes the observations incrementally
    def run(self):
        self.stop_flag = False  # This is done to avoid unexpected behavior
        print("[DEBUG] " + self.__class__.__name__ + " thread is running in background.")
        while not self.stop_flag:
            # First of all, it checkes if LowLevel didn't declare a failure
            found = self.tq.was_goal_inferred()
            if found == "failure":
                self.observations = []
                continue
            # Retrieves a new observation, when available
            observation = self.tq.get()  # Blocking call: if IntentionReading is not producing, HighLevel will pend here
            print("[DEBUG][HL] Read " + str(observation) + " from transition queue")
            self.observations.append(observation)
            # Decodes all the observations acquired
            goal = self.incremental_decode()
            # As soon as it is able to infer a goal, write it down
            if goal is not None:
                print("[DEBUG] Goal \'" + str(goal) + "\' was inferred. Trying to predict future emissions...")
                # Try to predict future emissions
                future_observations = self.predict_future_observations(self.observations)
                if future_observations is not None:     # If a prediction was possible, try a new inference
                    self.observations.extend(future_observations)
                    goal2 = self.incremental_decode()
                    if goal2 is not None and goal2 != goal:
                        print("[DEBUG] A new goal " + str(goal2) + " was predicted using cognitive prediction.")
                        # If a new goal was actually predicted, use this one, otherwise stick to the previous one
                        goal = goal2
                else:
                    print("[DEBUG] Future prediction didn't help. Retaining the previous inference.")
                print("[DEBUG] Selected goal: " + str(goal))
                self.tq.write_goal_name(goal)
                # Reset all observations done until that point
                self.observations = []
        print("[DEBUG] Shutting down " + self.__class__.__name__ + " thread.")

    # Sequence matcher: aids the decision process by trying to generate future emissions
    def predict_future_observations(self, observation):
        for l in range(1, len(observation) + 1):
            partial_obs = observation[:l]
            matching = []
            # Slices the lists and compares them
            for key, value in self.library.items():
                partial_value = value[:l]
                if partial_obs == partial_value:
                    matching.append(key)
            matches = len(matching)
            if matches == 1:  # Sequence was matched with no ambiguity
                return self.library[matching[0]][len(observation):]
            elif matches == 0:  # No entry in the library matched the sequence
                return None
        return None  # End of the sequence, ambiguity was not resolved
