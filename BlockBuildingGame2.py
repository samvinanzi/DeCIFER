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
from belief.face_vision import FaceVision
from BlockObserver import BlockObserver
from util.batch_simulator import sim

HELPER_CSV = "belief/datasets/examples/helper.csv"
TRICKER_CSV = "belief/datasets/examples/tricker.csv"


class BlockBuildingGame2:
    def __init__(self, debug=False, save=False, simulation=False, trust=False):
        self.cognition = CognitiveArchitecture(debug=debug, persist=save, enable_trust=trust, simulation=simulation)
        self.coordinates = robot.coordinates
        FaceVision.prepare_workspace()
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
        for goal in self.goals:
            self.goals[goal] = list(goal)
        self.debug = debug
        self.simulation = simulation
        self.trust_enabled = trust
        robot.action_home()
        robot.action_look(self.coordinates["center"])
        robot.say("Welcome to the intention reading experiment. We will start very soon.")
        time.sleep(2)

    # Main execution of the experiment
    def execute(self):
        if self.training_phase():   # Checks that the training is completed successfully
            if not self.trust_enabled:
                self.playing_phase_notrust()
            else:
                self.playing_phase_trust()
        else:
            robot.say("I'm sorry, something went wrong during training. Shall we try again?")
        self.end()

    # Terminates the experiment
    def end(self):
        robot.say("Thank you for playing!")
        self.cognition.terminate()      # Stops running threads, closes YARP ports, prints the recorded data log
        if self.simulation:
            sim.reset()

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

    # Robot and human will play the trust-aware game
    # NOTE! Setting automatic to True launches the batch simulator!
    def playing_phase_trust(self, point=False, automatic=True):
        trust_values = []
        correct_counter = 0 # counter
        turn_number = 1
        # The robot first recognizes the informer
        if not self.simulation:
            informer_id = self.cognition.trust.face_recognition()
        elif automatic:
            informer_id = 0
        else:
            informer_id = int(input("Informant ID (max " + str(self.cognition.trust.informants-1) + "): "))
        robot.say("Time to play! Feel free to start.")
        while True:
            if robot.__class__.__name__ == "Sawyer" and not self.simulation:
                robot.action_display("eyes")
            # The robot will try to understand the goal in progress
            goal = self.cognition.read_intention()
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            elif goal == "failure":
                robot.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                # This case is not managed because it represents an error (e.g. human standing still too long)
            else:
                # If not unknown, perform an action
                robot.say("I think we are building: " + goal)
                # A priori trust estimation (PROACTIVE)
                priori_trust, reliability = self.cognition.trust.beliefs[informer_id].is_informant_trustable()
                if self.debug:
                    print("[DEBUG] Informant " + str(informer_id) + " has reliability: " + str(reliability))
                # Explainability
                if priori_trust:
                    robot.say("I feel like I can trust you. I'll help you build this construction.")
                else:
                    robot.say("I don't think I can trust you. I'll complete this for you.")
                self.collaborate_with_trust(goal, priori_trust, point)
                # The robot will now evaluate the construction
                # Note that the robot might have built itself the construction because of a lack of trust
                if not priori_trust:
                    user_contruction = goal
                    correct = True
                    if automatic:
                        sim.exposed_goals.pop(0)    # We still have to pop one goal out
                else:
                    if not self.simulation:
                        user_contruction, correct = robot.evaluate_construction(goal)
                    else:
                        if not automatic:
                            user_contruction = input('Enter user construction: ').upper()
                        else:
                            user_contruction = sim.exposed_goals.pop(0)
                            print(user_contruction)
                        obs = BlockObserver()
                        obs.label = user_contruction
                        correct = obs.validate_sequence()
                self.cognition.record_outcome(correct)
                if correct:
                    correct_counter += 1
                # Explainability
                if correct:
                    phrase = "The construction is a valid one"
                    if user_contruction == goal:
                        phrase += " and I'm happy to have helped."
                    else:
                        phrase += ", but I misunderstood your intention. Sorry."
                    robot.say(phrase)
                else:
                    robot.say("This construction breaks the rules.")
                # Trust update, based on the outcome
                delta_trust = self.cognition.trust.update_trust(informer_id, correct, not priori_trust)
                # Log the new values
                _, new_trust = self.cognition.trust.beliefs[informer_id].is_informant_trustable()
                self.cognition.trust.update_log(new_trust)
                trust_values.append(new_trust)
                # Explainability: notify the user if the trust evaluation has changed
                if delta_trust == 1:
                    # User has gained trust
                    robot.say("You have proved yourself trustworthy.")
                elif delta_trust == -1:
                    # User has lost trust
                    robot.say("I'm sorry, but I don't trust you anymore.")
                if not correct:
                    # A posteriori trust estimation (REACTIVE)
                    posteriori_trust, reliability = self.cognition.trust.beliefs[informer_id].is_informant_trustable()
                    if posteriori_trust:
                        robot.say("I can't recognize this structure, but I trust you know what you are doing.")
                        # Give the user the option to teach a new goal
                        if not self.simulation:
                            self.ask_for_update()
                        else:
                            pass # todo
                    else:
                        # Explain the error
                        robot.say("We were building " + goal + ", but you built " + user_contruction + " instead.")
                #self.cognition.trust.update_log(reliability)
                #trust_values.append(reliability)
                # Asks the partner if to continue the game (only if task is not unknown)
                robot.say("Do you wish to continue with turn number " + str(turn_number + 1) + "?")
                if automatic:
                    response = "yes" if len(sim.exposed_goals) > 0 else "no"
                elif self.debug:
                    response = robot.wait_and_listen_dummy()
                else:
                    response = robot.wait_and_listen()
                if response != "yes":
                    break
                else:
                    robot.say("Ok, go!")
                    turn_number += 1
        # Success rate
        success_rate = correct_counter / len(trust_values)
        print("@@@ SUCCESS RATE: " + str(success_rate) + " @@@")
        return success_rate # NOTE: this will not allow plotting
        # Plot it
        import matplotlib.pyplot as plt
        import numpy as np
        plt.axhline(y=0, color='k', linestyle='-')
        plt.plot(np.arange(1, len(trust_values)+1, 1), trust_values)
        plt.xlabel("Turn")
        plt.ylabel("Trust")
        plt.title('Trust dynamics')
        plt.show()

    # Evaluates the output of a partial or total construction task
    def evaluate_output(self, informer_id, partial=False):
        # Evaluates the output
        user_contruction = sim.exposed_goals.pop(0)
        if self.debug:
            print(('Partial' if partial else 'Total') + " output observed:" + str(user_contruction))
        obs = BlockObserver()
        obs.label = user_contruction
        if partial:
            correct = obs.validate_partial_sequence()
        else:
            correct = obs.validate_sequence()
        # Trust update, based on the outcome
        delta_trust = self.cognition.trust.update_trust_r1(informer_id, correct)
        # Log the new values
        _, new_trust = self.cognition.trust.beliefs[informer_id].is_informant_trustable()
        self.cognition.trust.update_log(new_trust)
        # Explainability: notify the user if the trust evaluation has changed
        if delta_trust == 1:
            # User has gained trust
            robot.say("You have proved yourself trustworthy.")
        elif delta_trust == -1:
            # User has lost trust
            robot.say("I'm sorry, but I don't trust you anymore.")
        return user_contruction, correct, new_trust


    # Robot and human will play the trust-aware game
    # NOTE! Setting automatic to True launches the batch simulator!
    def playing_phase_trust_r1_simulation(self, point=False):
        trust_values = []
        correct_counter = 0  # counter
        turn_number = 1
        # The robot first recognizes the informer
        informer_id = 0
        robot.say("Time to play! Feel free to start.")
        while True:
            print("Turn " + str(turn_number))
            # The robot will try to understand the goal in progress
            goal = self.cognition.read_intention()
            # Acting, based on the intention read
            if goal == "unknown":  # If unknown, just wait until some prediction is made skipping all the rest
                continue
            elif goal == "failure":
                robot.say("I'm sorry, I can't understand what you are doing. I can't help you.")
                # This case is not managed because it represents an error (e.g. human standing still too long)
            else:
                # If not unknown, perform an action
                robot.say("I think we are building: " + goal)
                # The robot evaluates trust
                trust, reliability = self.cognition.trust.beliefs[informer_id].is_informant_trustable()
                if self.debug:
                    print("[DEBUG] Informant " + str(informer_id) + " has reliability: " + str(reliability))
                # TRUST SCENARIO
                if trust:
                    # The robot trusts the human. It will pass blocks to them and evaluate the total output
                    robot.say("I feel like I can trust you. I'll help you build this construction.") # Explainability
                    #self.collaborate_with_trust(goal, trust, point) # Passes the blocks
                    if self.debug:
                        print("Robot is passing the remaining blocks to the user")
                    # Evaluates the total output
                    user_contruction, correct, new_trust = self.evaluate_output(informer_id, partial=False)
                    trust_values.append(new_trust)
                    if correct:
                        correct_counter += 1
                        phrase = "The construction is a valid one"
                        if user_contruction == goal:
                            phrase += " and I'm happy to have helped."
                        else:
                            phrase += ", but I misunderstood your intention. Sorry."
                        robot.say(phrase)
                    else:
                        robot.say("This construction breaks the rules.")
                    self.cognition.record_outcome(correct)
                # DISTRUST SCENARIO
                else:
                    # The robot doesn't trust the human. It will inspect the partial output.
                    user_contruction, correct, new_trust = self.evaluate_output(informer_id, partial=True)
                    trust_values.append(new_trust)
                    # Robot acting
                    if correct:
                        if self.debug:
                            print("[DEBUG] robot is passing the blocks to the human")
                        pass    # todo implement
                    else:
                        # Explain the error
                        robot.say("We were building " + goal + ", but you built " + user_contruction + " instead.")
                        if self.debug:
                            print("[DEBUG] robot is fixing the wrong construction")     # todo implement
                    correct_counter += 1
                    self.cognition.record_outcome(True)
                robot.say("Do you wish to continue with turn number " + str(turn_number + 1) + "?")
                response = "yes" if len(sim.exposed_goals) > 0 else "no"
                if response != "yes":
                    break
                else:
                    robot.say("Ok, go!")
                    turn_number += 1
        # Success rate
        success_rate = correct_counter / len(trust_values)
        print("@@@ SUCCESS RATE: " + str(success_rate) + " @@@")
        return success_rate     # Comment out to display the graph

        # Plot it
        import matplotlib.pyplot as plt
        import numpy as np
        plt.axhline(y=0, color='k', linestyle='-')
        plt.plot(np.arange(1, len(trust_values) + 1, 1), trust_values)
        plt.xlabel("Turn")
        plt.ylabel("Trust")
        plt.title('Trust dynamics')
        plt.show()

    # Proactive collaboration. If the user is trusted the robot provides the blocks, otherwise it also places them
    def collaborate_with_trust(self, goal, priori_trust, point=False):
        # Determines the next blocks to be collected
        collected_blocks = robot.count_blocks()
        remaining_blocks = self.goals[goal][collected_blocks:]
        if priori_trust:
            # Pass the blocks to the user
            for block in remaining_blocks:
                if not self.simulation:
                    self.interact_with_single_block(block, grasp=not point)
                else:
                    print("Robot is passing block " + str(block) + " to the user.")
                    #time.sleep(1)
        else:
            # Positions the blocks in place itself
            for block in remaining_blocks:
                # Determine the position for that block (reversed for the robot) := [4] [3] [2] [1]
                position = self.goals[goal].index(block) + 1
                if not self.simulation:
                    self.collect_and_place(block, position)
                else:
                    print("Robot is positioning block " + str(block) + " in position " + str(position) + ".")
                    #time.sleep(1)

    # Collects a block from the table and places it in the correct order
    def collect_and_place(self, block, position):
        coordinates = robot.block_coordinates[block]
        robot.action_take(coordinates)
        robot.action_position_block(block, position)
        #time.sleep(1)
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
        robot.say("Do you want to teach me a new goal?")
        if self.debug:
            response = robot.wait_and_listen_dummy()
        else:
            response = robot.wait_and_listen()
        if response == "yes":
            self.cognition.update()  # Undergo a new training
            robot.say("I have learned the new goal. Let's continue!")
        return True if response == "yes" else False
