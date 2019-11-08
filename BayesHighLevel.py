"""

New High-Level, using Bayesian Networks instead of a Hidden Semi-Markov Model.
The network is formed by an Intention node and N observation nodes.

[I]---------
 |         |
 V         V
[O1] --> [O2]

"""

from pomegranate import *
from StopThread import StopThread


class HighLevel(StopThread):
    def __init__(self, transition_queue):
        StopThread.__init__(self)
        self.model = BayesianNetwork("High-Level Intention Reading")
        self.data = {}
        self.symbols = []
        self.goals = []
        self.length = None
        self.internals = {'intention': None, 'observations': {}}
        self.tq = transition_queue
        self.observations = []

    # Builds the model according to the data
    def build_model(self, data):
        # Convert the training data to a suitable format
        for dict_entry in data:
            label = dict_entry['label']
            values = dict_entry['data']
            self.data[label] = values
        self.initialize()
        self.make_model_prefixed()
        self.fit()

    def initialize(self):
        for key, value in self.data.items():
            self.goals.append(key)
            self.symbols.append(value)
            self.length = len(value)
        self.symbols = list(set([item for sublist in self.symbols for item in sublist]))    # Set of possible symbols

    def build_intention_node(self):
        output = {}
        for key, _ in self.data.items():
            output[key] = 1. / len(self.data)
        return output

    def build_nth_node(self, n=0):
        output = []
        for h in self.goals:
            for e in self.symbols:
                if n == 0:
                    output.append([h, e, 0])
                else:
                    for s in self.symbols:
                        output.append([h, e, s, 0])
        return output

    # todo not working
    # Dynamically creates the network with a variable number of observation nodes
    def make_model(self):
        # Create the intention node
        intention = DiscreteDistribution(self.build_intention_node())
        self.internals['intention'] = intention
        # Create the variable-lenght observation nodes
        for i in range(self.length):
            parents = [intention]
            if i > 0:
                parents.append(self.internals['observations'][i-1])
            obs = ConditionalProbabilityTable(self.build_nth_node(i), parents)
            self.internals['observations'][i] = obs
        # Prepare the states
        intention_state = State(intention, name="intention")
        print("Adding the " + intention_state.name + " node")
        self.model.add_state(intention_state)
        states = []
        for key, value in sorted(self.internals['observations'].items()):
            new_state = State(value, name="observation "+str(key))
            print("Adding the observation " + str(key) + " node.")
            self.model.add_state(new_state)
            states.append(new_state)
        # Define the edges and bake
        for i in range(1, len(states)):
            print("Adding edge from " + intention_state.name + " to " + states[i].name)
            self.model.add_edge(intention_state, states[i])
            print("Adding edge from " + states[i-1].name + " to " + states[i].name)
            self.model.add_edge(states[i-1], states[i])
        print("Baking")
        self.model.bake()
        print("Plotting")
        self.model.plot()

    # Builds the network. Hard-coded.
    def make_model_prefixed(self):
        # Probability tables
        intention = DiscreteDistribution(self.build_intention_node())
        obs1 = ConditionalProbabilityTable(self.build_nth_node(0), [intention])
        obs2 = ConditionalProbabilityTable(self.build_nth_node(1), [intention, obs1])
        obs3 = ConditionalProbabilityTable(self.build_nth_node(2), [intention, obs2])
        obs4 = ConditionalProbabilityTable(self.build_nth_node(3), [intention, obs3])
        # States (a.k.a. nodes)
        i = State(intention, name="intention")
        o1 = State(obs1, name="observation 1")
        o2 = State(obs2, name="observation 2")
        o3 = State(obs3, name="observation 3")
        o4 = State(obs4, name="observation 4")
        self.model.add_nodes(i, o1, o2, o3, o4)
        # Edges
        self.model.add_edge(i, o1)
        self.model.add_edge(i, o2)
        self.model.add_edge(i, o3)
        self.model.add_edge(i, o4)
        self.model.add_edge(o1, o2)
        self.model.add_edge(o2, o3)
        self.model.add_edge(o3, o4)
        # Finalize
        self.model.bake()

    # Transforms the building data into fitting data
    def get_training_data(self):
        # Converts data in training format
        fit_data = []
        for key, value in self.data.items():
            new_data = [key]
            for item in value:
                new_data.append(item)
            fit_data.append(new_data)
        return fit_data

    # Fits the model to the training data
    def fit(self):
        data = self.get_training_data()
        self.model.fit(data, pseudocount=0.01)

    # Performs message passing algorithm to infer the probability of the intention given (partial) evidence.
    # It returns the most probable label and a confidence value, if it exists.
    def predict_state(self, evidence):
        probabilities = self.model.predict_proba(evidence)[0].parameters[0]
        goal = None
        score = -INF
        tie = False
        for key, value in probabilities.items():
            if value == score:
                tie = True
            if value > score:
                score = value
                goal = key
                tie = False
        if not tie:
            return goal
        else:
            return None

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
            # Builds up the evidence it has, as: [intention (None), observation1, ... , observationN]
            evidence = [None]
            for i in range(self.length):
                try:
                    observation = self.observations[i]
                except IndexError:
                    observation = None
                finally:
                    evidence.append(observation)
            # Decodes all the observations acquired
            goal = self.predict_state(evidence)
            if goal is not None:
                # As soon as it is able to infer a goal, write it down
                print("[DEBUG] Current inferred goal is: " + str(goal))
                self.tq.write_goal_name(goal)
                # Reset all observations done until that point
                self.observations = []
        print("[DEBUG] Shutting down " + self.__class__.__name__ + " thread.")
