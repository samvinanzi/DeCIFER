"""

This class handles trust and belief from the robot side.

"""

from belief.bayesianNetwork import BeliefNetwork
from belief.datasetParser import DatasetParser
from belief.episode import Episode
from belief.face_vision import FaceVision, TrainingData
from robots.robot_selector import robot
import cv2
import os


class Trust:
    def __init__(self):
        self.training_data = TrainingData()
        self.informants = 0
        self.beliefs = []
        self.time = None
        self.face_frames_captured = 10
        self.load_time()

    # --- FACE DETECTION AND RECOGNITION ---

    # If image contains a face, it retrieves the cropped region of interest
    def detect_face(self, image, grayscale=True):
        roi = FaceVision.facial_detection(image, grayscale=grayscale)
        return False if roi is None else True, roi

    # Captures a certain amount of face frames
    def collect_face_frames(self, number):
        face_frames = []
        found_faces = 0
        undetected_frames = 0
        while found_faces < number:
            image = robot.get_camera_frame()
            detected, roi = self.detect_face(image)
            if detected:
                found_faces += 1
                face_frames.append(roi)
                undetected_frames = 0
            else:
                undetected_frames += 1
                if undetected_frames % 10 == 0:
                    robot.say("I can't see you well. Can you please move closer?")
            #cv2.putText(image, str("Detected: " + str(found_faces) + " / " + str(number)),
            #            (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            #cv2.imshow("Robot Eyes", image)
            #cv2.waitKey(1)
        return face_frames

    # Obtains training samples of one of the informers
    # Automatically updates the informant number
    # Saves the frames in the captures directory
    def acquire_examples(self, number_of_frames, informant_number):
        robot.say("Hello informer number " + str(informant_number) + ". Please look at me")
        frames = self.collect_face_frames(number_of_frames)
        robot.say("Thank you")
        count = 1
        for frame in frames:
            self.training_data.images.append(frame)
            self.training_data.labels.append(informant_number)
            cv2.imwrite("captures/" + str(informant_number) + "-" + str(count) + ".jpg", frame)
            count += 1
        self.informants += 1

    # Finalizes learning by training the model with all the data acquired
    def face_learning(self):
        FaceVision.recognition_train(self.training_data.prepare_for_training())

    # Recognizes a face
    # Collects an amount of frames, gets a prediction on each of them and returns the most predicted label
    def face_recognition(self, number_of_frames=5, announce=True):
        unknown = False
        robot.say("Please look at me")
        # Collect face data
        frames = self.collect_face_frames(number_of_frames)
        # Counts the recognized labels
        predictions = [0 for i in range(self.informants+1)]  # An extra slot to consider the -1 (unknown informant) case
        for frame in frames:
            predictions[FaceVision.recognition_predict(frame)] += 1
        # Returns the maximum
        guess = predictions.index(max(predictions))
        # If the maximum value found is in the last position of the list, it's an unrecognized informant
        if guess == len(predictions)-1:
            unknown = True
            # Unknown informant! Adding it to the known ones and generating episodic memory
            self.manage_unknown_informant(frames)
            # This new informant has the biggest label yet
            guess = self.informants - 1
        if announce:
            if not unknown:
                robot.say("Hello again, informer " + str(guess))
            else:
                robot.say("I've never seen you before, I'll call you informer " + str(guess))
        return guess

    # Manages the unknown informant detection
    def manage_unknown_informant(self, frames):
        # Updates the model with the acquired frames and the right label
        new_data = TrainingData()
        new_data.images = frames
        new_data.labels = [self.informants for i in range(len(frames))]
        FaceVision.recognition_update(new_data.prepare_for_training())
        # Creates an episodic belief network
        name = "Informer" + str(self.informants)
        episodic_network = BeliefNetwork.create_episodic(self.beliefs, self.get_and_inc_time(), name=name)
        self.beliefs.append(episodic_network)
        # Updates the total of known informants
        self.informants += 1    # This is done at the end because the label for the class is actually self.informants-1

    # --- TIME (episodic memory) ---

    # Load time value from file
    def load_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'r') as f:
                self.time = int(f.readline())
        else:
            self.time = 0

    # Increases and saves the current time value
    def get_and_inc_time(self):
        previous_time = self.time
        self.time += 1
        with open("current_time.csv", 'w') as f:
            f.write(str(self.time))
        return previous_time

    # Reset time
    def reset_time(self):
        if os.path.isfile("current_time.csv"):
            with open("current_time.csv", 'w') as f:
                f.write("0")
        self.time = 0

    # --- TRUST MODEL SAVE/LOAD ---

    # Saves the beliefs
    def save_beliefs(self):
        if not os.path.exists("datasets/"):
            os.makedirs("datasets/")
        for belief in self.beliefs:
            belief.save()

    # Loads the beliefs
    def load_beliefs(self, path="datasets/"):
        # Resets previous beliefs
        self.beliefs = []
        i = 0
        while os.path.isfile(path + "Informer" + str(i) + ".csv"):
            self.beliefs.append(BeliefNetwork("Informer" + str(i), path + "Informer" + str(i) + ".csv"))
            i += 1

    # --- TRUST MANIPULATION ---

    # Updates the trust in the user with two symmetrical positive or negative examples
    def update_trust(self, informant_id, correctness):
        assert 0 <= informant_id < self.informants, "Invalid informant_id argument"
        assert isinstance(correctness, bool), "Correctness argument must be boolean"
        if correctness:
            new_evidence = Episode.create_positive()
        else:
            new_evidence = Episode.create_negative()
        self.beliefs[informant_id].update_belief(new_evidence)  # updates belief with correct or wrong episode
        self.beliefs[informant_id].update_belief(new_evidence.generate_symmetric())  # symmetric episode is generated

    # In the experiment, the trainer must be trusted automatically
    # Equivalent to a Vanderbilt familiarization phase with one trustable informant
    def learn_and_trust_trainer(self):
        self.acquire_examples(self.face_frames_captured, 0)
        self.beliefs.append(BeliefNetwork("Informer 0", Episode.create_positive()))
        self.face_learning()
