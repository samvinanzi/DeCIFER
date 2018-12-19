"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

Version 2 with trust evaluations.

"""

from CognitiveArchitecture import CognitiveArchitecture
from iCub import icub
from belief.bayesianNetwork import BeliefNetwork
from belief.episode import Episode
import time
from collections import Counter
from Construction import Shape, Construction


class BlockBuildingGame:
    def __init__(self, debug=False, fixed_goal=True):
        self.cognition = CognitiveArchitecture()
        self.coordinates = {
            "left": (-1.0, -0.5, -0.5),
            "right": (-1.0, 0.5, -0.5),
            "center": (0.0, 0.0, 0.0)
        }
        self.goals = {      # Direction sequences for each goal
            "tower": [],
            "wall": [],
            "castle": [],
            "clean": []
        }
        self.debug = debug
        # todo move it in Robot class after experiment 1 is completed
        self.bbn = BeliefNetwork("iCub_belief", "belief/datasets/examples/helper.csv")  # Trusting belief network
        self.colors = {         # Position and name of the two block colors
            "left": "blue",
            "right": "red"
        }
        self.constructions = {}     # The details (shape and color) of the correct block constructions
        self.fixed_goal = fixed_goal        # Is the experiment in fixed or mutable goal configuration?
        # Bring iCub in home position
        icub.action_home()

    # Main execution of the experiment
    def execute(self):
        if self.training_phase():
            self.playing_phase()
        else:
            icub.say("I'm sorry, something went wrong. Shall we try again?")
        self.end()

    # Terminates the experiment
    def end(self):
        icub.say("Thank you for playing!")
        self.cognition.terminate()      # Stops running threads, closes YARP ports, prints the recorded data log

    # Trains the robot on the current rules of the game
    def training_phase(self):
        icub.say("Show me the rules of the game, I will learn them so that we can play together.")
        self.cognition.train()
        # Checks that all the four goals have been learned
        for intention in self.cognition.lowlevel.train.intentions:
            if intention.goal not in self.goals:
                print("Goal " + intention.goal + " not included in the ones for this experiment! Please re-train.")
                return False
            else:
                print("Goal " + intention.goal + " correctly learned.")
        # Now the robot needs to learn the directions of movement
        cluster_orientations = self.cognition.lowlevel.train.cluster_orientation_reach()
        # Associate left / right movements for each goal
        for intention in self.cognition.lowlevel.train.intentions:
            for cluster in intention.actions:
                self.goals[intention.goal].append(cluster_orientations[cluster])
        self.get_target_constructions()     # Stores the correct contructions details
        return True

    # Stores the expected constructions, based on the game and how the rules were taught (the kind of constructions and
    # their shape is fixed, but the colors are mutable and depend on the training received from the experimenter)
    def get_target_constructions(self):
        for goal in ["tower", "wall", "castle"]:  # The goals associated with building a construction
            directions = self.get_directions_sequence(goal)
            if goal == "tower":
                shape = Shape.VERTICAL_RECT
            elif goal == "wall":
                shape = Shape.HORIZONTAL_RECT
            else:
                shape = Shape.SQUARE
            reds = [self.colors[x] for x in directions].count("red")
            blues = [self.colors[x] for x in directions].count("blue")
            self.constructions[goal] = Construction(shape, reds, blues)

    # Robot and human partner will play the game cooperatively
    def playing_phase(self, unrecognized_limit=5):
        icub.say("Ok, time to play! Feel free to start.")
        unrecognized_actions = 0
        while True:
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, wait
                unrecognized_actions += 1
                if unrecognized_actions >= unrecognized_limit:
                    icub.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                    icub.say("Do you wish to teach me something new?")
                    self.ask_for_update()
                else:
                    continue    # Start again from the beginning of the loop (skipping the request for continuation)
            else:
                # If not unknown, perform an action
                unrecognized_actions = 0
                if goal == "clean":
                    self.put_away()
                else:       # Tower, wall or castle
                    trust = self.bbn.is_informant_trustable()           # A priori trust estimation
                    if not trust:
                        self.give_advice(goal)
                    self.collect_blocks(goal)                           # Task execution
                    correctness = self.evaluate_construction(goal)      # A posteriori trust update
                    if not self.fixed_goal and not trust and not correctness:
                        icub.say("I've noticed that you keep performing actions I can't understand. \
                        Do you wish to train me on them?")
                        self.ask_for_update()
            # Asks the partner if to continue the game (only if task is not unknown)
            icub.say("Do you wish to continue?")
            if self.debug:
                response = icub.wait_and_listen_dummy()
            else:
                response = icub.wait_and_listen()
            if response != "yes":
                break

    # Gives advice to an untrustable informant
    def give_advice(self, goal):
        icub.say("I just want to be sure you remember the rules.")
        direction_sequence = self.get_directions_sequence(self.goals[goal])
        string = "Are you building a " + goal + "? In that case, you'll need: "
        # Counts the occurencies of left/right for the inferred goal, associates them to colors, orders them and prints
        # them in a readeable (speakable) format.
        count = Counter(direction_sequence)
        for i, duple in enumerate(count.most_common(2)):
            if i == 1:
                string += " and "
            string += str(duple[1]) + " " + self.colors[duple[0]] + ("blocks" if duple[1] > 1 else "block")
        if goal == "tower":
            icub.say("You must place them in a column shape.")
        elif goal == "wall":
            icub.say("You must place them in a row shape.")
        else:
            icub.say("You must place them to form a triangle.")

    # Determines if the building was constructed correctly and updates the robot's belief
    # Returns true or false based on the evaluation
    def evaluate_construction(self, goal):
        construction = icub.observe_for_shape_and_color()   # analyzes the built construction
        if construction == self.constructions[goal]:        # compares it to the target one
            new_evidence = Episode([0, 0, 0, 0])
            correct = True
        else:
            new_evidence = Episode([1, 1, 1, 1])
            correct = False
        self.bbn.update_belief(new_evidence)                        # updates belief with correct or wrong episode
        self.bbn.update_belief(new_evidence.generate_symmetric())   # symmetric episode is generated too
        return correct

    # Ask the partner if he or she desires to update the knowledge base of the robot
    def ask_for_update(self):
        if self.debug:
            response = icub.wait_and_listen_dummy()
        else:
            response = icub.wait_and_listen()
        if response == "yes":
            self.cognition.update()     # Undergo a new training
        return True if response == "yes" else False

    # Determines where to obtain the blocks from
    def get_directions_sequence(self, goal):
        transitions = self.goals[goal]
        # Filters out the center positions to obtain a sequence of only lefts and rights
        return list(filter(lambda x: x != "center", transitions))   # previously was [1:]

    # Looks to one side, seeks for a cube, picks it up and gives it to the human partner
    # If the latter doesn't pick it up, puts it down and picks a different colored one
    def collect_single_block(self, direction, wait_time=5):
        icub.action_look(self.coordinates[direction])
        while True:
            object_centroid = icub.observe_for_centroid()
            if object_centroid is None:
                icub.say("I can't see any objects...")
                time.sleep(2)
            else:
                break
        object_coordinates = icub.get_object_coordinates(list(object_centroid))
        if icub.action_take(object_coordinates):
            icub.action_give()
            # Gives some time to the partner to collect the block from its hand
            delay_time = 0
            while delay_time < wait_time:
                if icub.is_holding():
                    delay_time += 1
                    time.sleep(1)
                else:
                    break
            if delay_time >= wait_time:
                # The user is not collecting the block
                if direction != "center":       # This should always be true, just a sanity control
                    icub.say("Maybe I was wrong and you need a different colored block.")
                else:
                    icub.say("Maybe you don't need this block.")
                # Puts away the block
                icub.action_drop(self.coordinates[direction])
                # Swaps the direction
                if direction == "left":
                    self.collect_single_block("right")
                elif direction == "right":
                    self.collect_single_block("left")
            else:   # if action_take fails
                icub.say("Oh my, I wasn't able to grasp it. Let me try again.")
        icub.action_home()

    # Collects the blocks, in the order provided by the direction sequence
    def collect_blocks(self, goal):
        direction_sequence = self.get_directions_sequence(goal)
        completed_steps = icub.count_objects()
        for i in range(completed_steps, len(direction_sequence)):
            self.collect_single_block(direction_sequence[i])
            time.sleep(2)

    # Receives a cube and puts it down in the toy chest
    def put_away(self, wait_time=5):
        blocks = icub.count_objects()
        for i in range(0, blocks):
            icub.action_expect()
            delay_time = 0
            while delay_time < wait_time:   # Waits for its hand to be loaded, or cancels its action
                if not icub.is_holding():
                    delay_time += 1
                    time.sleep(1)
                else:
                    icub.action_drop(self.coordinates['center'])
                    break
            icub.action_home()
