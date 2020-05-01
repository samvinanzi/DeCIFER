"""

Block building game to test the cognitive architecture. The robot will learn the rules of the game and then help its
partner to build one of the three constructions: towers, walls and castles.
Tested for Sawyer.
Note (for myself): the cognitive system is generic and can learn whichever goals, this experiment is specific.

Version 2 with trust estimation.

"""

from CognitiveArchitecture import CognitiveArchitecture
import time
from robots.robot_selector import robot
from belief.episode import Episode
from belief.bayesianNetwork import BeliefNetwork

HELPER_CSV = "belief/datasets/examples/helper.csv"
TRICKER_CSV = "belief/datasets/examples/tricker.csv"


class BlockBuildingGame2:
    def __init__(self, debug=False, save=False, fixed_goal=True, naive_trust=True):
        self.cognition = CognitiveArchitecture(debug=debug, persist=save)
        self.coordinates = robot.coordinates
        if naive_trust:
            self.bbn = BeliefNetwork("partner", HELPER_CSV)
        else:
            self.bbn = BeliefNetwork("partner", TRICKER_CSV)
        self.goals = {
            "BROG": [],
            "BGOR": [],
            "ORBG": [],
            "OGBR": [],
            "GBRO": [],
            "GORB": [],
            "RBGO": [],
            "ROGB": [],
        }
        self.debug = debug
        self.fixed_goal = fixed_goal  # Is the experiment in fixed or mutable goal configuration?
        robot.action_home()
        robot.action_look(self.coordinates["center"])
        robot.say("Welcome to the intention reading experiment. We will start very soon.")
        time.sleep(2)

    # Main execution of the experiment
    def execute(self):
        if self.training_phase():
            self.playing_phase()
        else:
            robot.say("I'm sorry, something went wrong. Shall we try again?")
        self.end()

    # Terminates the experiment
    def end(self):
        robot.say("Thank you for playing!")
        self.cognition.terminate()      # Stops running threads, closes YARP ports, prints the recorded data log

    # Trains the robot on the current rules of the game
    def training_phase(self):
        robot.say("Show me the rules of the game, I will learn them so that we can play together.")
        self.cognition.train()
        # Checks that all the four goals have been learned
        for intention in self.cognition.lowlevel.train.intentions:
            if intention.goal not in self.goals:
                print("Goal " + intention.goal + " not included in the ones for this experiment! Please re-train.")
                return False
            else:
                print("Goal " + intention.goal + " correctly learned.")
        return True

    # Reloads a previous training, to be able to play immediately
    def reload_training(self):
        robot.say("Let me remember the rules of the game...")
        self.cognition.train(reload=True)
        if robot.AUTORESPONDER_ENABLED:
            robot.command_list = robot.command_list[0:-8]     # Truncates the automated response sequence
        robot.say("Ok, done!")

    # Debug mode
    def debug_human(self):
        robot.say("Ready to start debugging!")
        self.cognition.read_intention(debug=True)

    """
    # Robot and human partner will play the game cooperatively
    def playing_phase(self, point=False):
        turn_number = 1
        robot.say("Time to play! Feel free to start.")
        while True:
            if robot.__class__.__name__ == "Sawyer":
                robot.action_display("eyes")
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            elif goal == "failure":
                robot.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                if not self.fixed_goal:
                    robot.say("Do you wish to teach me something new?")
                    self.ask_for_update()
            else:
                # If not unknown, perform an action
                robot.say("We are building a " + goal)
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
    """

    # Robot and human will play the trust-aware game
    def playing_phase_trust(self, point=False):
        turn_number = 1
        robot.say("Time to play! Feel free to start.")
        while True:
            if robot.__class__.__name__ == "Sawyer":
                robot.action_display("eyes")
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            elif goal == "failure":
                robot.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                # This case is not managed because it represents an error (e.g. human standing still too long)
            else:
                # If not unknown, perform an action
                robot.say("We are building a " + goal)
                priori_trust = self.bbn.is_informant_trustable()  # A priori trust estimation (PROACTIVE)
                self.collaborate_with_trust(goal, priori_trust, point)
                # The robot will now evaluate the construction
                user_contruction, correct = robot.evaluate_construction(goal)
                self.update_trust(correct)  # Trust update, based on the outcome
                if not correct:
                    posteriori_trust = self.bbn.is_informant_trustable()  # A posteriori trust estimation (REACTIVE)
                    if posteriori_trust:
                        # Give the user the option to teach a new goal
                        self.ask_for_update()
                    else:
                        # Explain the error
                        robot.say("We were building a " + goal + " but you built: " + user_contruction)
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

    # Proactive collaboration. If the user is trusted the robot provides the blocks, otherwise it also places them
    def collaborate_with_trust(self, goal, priori_trust, point=False):
        # Determines the next blocks to be collected
        collected_blocks = robot.count_blocks()
        remaining_blocks = self.goals[goal][collected_blocks:]
        if priori_trust:
            # Pass the blocks to the user
            for block in remaining_blocks:
                self.interact_with_single_block(block, grasp=not point)
        else:
            # Positions the blocks in place
            for block in remaining_blocks:
                self.collect_and_place(block)

    # Updates the trust in the user with two symmetrical positive or negative example
    def update_trust(self, correctness):
        assert isinstance(correctness, bool), "Correctness argument must be boolean"
        if correctness:
            new_evidence = Episode([0, 0, 0, 0])
        else:
            new_evidence = Episode([0, 0, 0, 1])
        self.bbn.update_belief(new_evidence)  # updates belief with correct or wrong episode
        self.bbn.update_belief(new_evidence.generate_symmetric())  # symmetric episode is generated too

    # Collects a block from the table and places it in the correct order
    def collect_and_place(self, block):
        coordinates = robot.block_coordinates[block]
        robot.action_take(coordinates)
        # todo: place on the building area
        time.sleep(1)
        robot.action_home()

    # Plays, but without trust evaluations. If demo is True, it skips the listening
    def playing_phase_notrust(self, point=False, demo=False):
        turn_number = 1
        robot.say("Time to play! Feel free to start.")
        while True:
            if robot.__class__.__name__ == "Sawyer":
                robot.action_display("eyes")
            goal = self.cognition.read_intention()  # The robot will try to understand the goal in progress
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            elif goal == "failure":
                robot.say("I'm sorry, I cannot understand what you are doing.")
            else:
                # If not unknown, perform an action
                robot.say("We are building a " + goal)
                self.collaborate(goal, point)
            # Asks the partner if to continue the game (only if task is not unknown)
            robot.say("Do you wish to continue with turn number " + str(turn_number + 1) + "?")
            if self.debug:
                response = robot.wait_and_listen_dummy()
            else:
                response = robot.wait_and_listen()
            if response != "yes":
                break
            else:
                time.sleep(1)
                robot.say("Ok, go!")
                turn_number += 1

    # Plays a demo. The robot will recognize a predefined sequence of intentions
    def play_demo_notrust(self):
        intentions = ["BGOR", "OGBR", "GBRO", "RBGO"]
        turn_number = 1
        robot.say("Time to play! Feel free to start.")
        for intention in intentions:
            robot.action_display("eyes")
            time.sleep(5)
            robot.say("We are building a " + intention)
            self.collaborate(intention)
            robot.say("Do you wish to continue with turn number " + str(turn_number + 1) + "?")
            time.sleep(1)
            if turn_number < 4:
                robot.say("Ok, go!")
                turn_number += 1
            else:
                break

    '''
    # Determines if the building was constructed correctly and updates the robot's belief
    # Returns true or false based on the evaluation
    def evaluate_construction(self, goal):
        correctness = robot.evaluate_construction()
        if correctness:
            new_evidence = Episode([0, 0, 0, 0])
        else:
            new_evidence = Episode([1, 1, 1, 0])
        self.bbn.update_belief(new_evidence)  # updates belief with correct or wrong episode
        self.bbn.update_belief(new_evidence.generate_symmetric())  # symmetric episode is generated too
        return correctness
    '''

    # Gives advice to an untrustable informant
    def give_advice(self, goal):
        robot.say("I just want to be sure you remember the rules.")
        robot.say("Are you building a " + goal + "? You'll need to build a row using this order of blocks:")
        robot.say(",".join([str(i) for i in self.goals[goal]]))

    # Perform collaborative behavior: collect of point the blocks
    def collaborate(self, goal, grasp=True):
        # Finds out which blocks are missing in the predicted structure
        collected_blocks = robot.count_blocks()
        remaining_blocks = self.goals[goal][collected_blocks:]
        for block in remaining_blocks:
            self.interact_with_single_block(block, grasp)
        robot.action_look(self.coordinates["center"])    # Look back at the user

    # Grasps or collects a single block
    def interact_with_single_block(self, color, grasp=True):
        coordinates = robot.block_coordinates[color]
        if grasp:
            robot.action_take(coordinates)
            robot.action_give()
        else:
            robot.action_point(coordinates)
        time.sleep(1)
        robot.action_home()

    # Ask the partner if he or she desires to update the knowledge base of the robot
    def ask_for_update(self):
        if self.debug:
            response = robot.wait_and_listen_dummy()
        else:
            response = robot.wait_and_listen()
        if response == "yes":
            self.cognition.update()  # Undergo a new training
        return True if response == "yes" else False
