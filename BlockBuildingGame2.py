"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

Version 2 with trust evaluations.

"""


from collections import Counter

import BlockBuildingGame
from Construction import Shape, Construction
from belief.episode import Episode
from robots.robot_selector import robot


class BlockBuildingGame2(BlockBuildingGame):
    def __init__(self, debug=False, fixed_goal=True):
        super().__init__(debug)
        self.constructions = {}  # The details (shape and color) of the correct block constructions
        self.fixed_goal = fixed_goal  # Is the experiment in fixed or mutable goal configuration?

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
    def playing_phase(self, unrecognized_limit=5, point=False):
        turn_number = 1
        unrecognized_actions = 0
        robot.say("Time to play! Feel free to start.")
        while True:
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                unrecognized_actions += 1
                if unrecognized_actions >= unrecognized_limit:
                    robot.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                    robot.say("Do you wish to teach me something new?")
                    self.ask_for_update()
                else:
                    continue  # Start again from the beginning of the loop (skipping the request for continuation)
            else:
                unrecognized_actions = 0
                robot.say("We are building a " + goal)
                # If not unknown, perform an action
                trust = self.bbn.is_informant_trustable()  # A priori trust estimation
                if not trust:
                    self.give_advice(goal)
                self.collaborate(goal, point)
                correctness = self.evaluate_construction(goal)  # A posteriori trust update
                if not self.fixed_goal and not trust and not correctness:
                    robot.say("I've noticed that you keep performing actions I can't understand. \
                                        Do you wish to train me on them?")
                    self.ask_for_update()
            # Asks the partner if to continue the game (only if task is not unknown)
            robot.say("Do you wish to continue with turn number " + str(turn_number + 1) + "?")
            if self.debug:
                response = robot.wait_and_listen_dummy()
            else:
                response = robot.wait_and_listen()
            if response != "yes":
                break
            else:
                robot.say("Ok, go!")
                turn_number += 1

    # Gives advice to an untrustable informant
    def give_advice(self, goal):
        robot.say("I just want to be sure you remember the rules.")
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
            robot.say("You must place them in a column shape.")
        elif goal == "wall":
            robot.say("You must place them in a row shape.")
        else:
            robot.say("You must place them to form a triangle.")

    # Determines if the building was constructed correctly and updates the robot's belief
    # Returns true or false based on the evaluation
    def evaluate_construction(self, goal):
        construction = robot.observe_for_shape_and_color()  # analyzes the built construction
        if construction == self.constructions[goal]:  # compares it to the target one
            new_evidence = Episode([0, 0, 0, 0])
            correct = True
        else:
            new_evidence = Episode([1, 1, 1, 1])
            correct = False
        self.bbn.update_belief(new_evidence)  # updates belief with correct or wrong episode
        self.bbn.update_belief(new_evidence.generate_symmetric())  # symmetric episode is generated too
        return correct

    # Ask the partner if he or she desires to update the knowledge base of the robot
    def ask_for_update(self):
        if self.debug:
            response = robot.wait_and_listen_dummy()
        else:
            response = robot.wait_and_listen()
        if response == "yes":
            self.cognition.update()  # Undergo a new training
        return True if response == "yes" else False

