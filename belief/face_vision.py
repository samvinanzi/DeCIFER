"""

Collection of scripts that perform face detection, memorization and recognition.

Methods included in this file manage the face recognition algorithms.
Selectable algorithms:  0: EigenFaces
                        1: FisherFaces
                        2: Local Binary Patterns Histograms (LBPH)
Recommended (and default) is LBPH, as it is the only one which supports updating. Other methods will need to run
a new training with all the old samples plus the new ones.

"""

import os
import cv2
import functools
import numpy as np


class FaceVision:
    MODEL_FILE = "../classifiers/robotvision.yml"       # Because this file is located in the belief/ subdirectory
    ALGORITHM_NUMBER = 2
    CAPTURES_DIR = "../captures/"

    # --- DETECTION ---

    # Creates the working directory or empties it
    @staticmethod
    def prepare_workspace():
        if not os.path.exists(FaceVision.CAPTURES_DIR):
            os.mkdir(FaceVision.CAPTURES_DIR)
        else:
            file_list = [f for f in os.listdir(FaceVision.CAPTURES_DIR)]
            for f in file_list:
                os.remove(os.path.join(FaceVision.CAPTURES_DIR, f))

    @staticmethod
    def facial_detection(img, scale_factor=1.4, min_neighbours=5, single=True, debug=False, grayscale=True):
        """ Performs facial detection within an image
        :param img: image data matrix
        :param scale_factor: how much the image size is reduced at each image scale
        :param min_neighbours: how many neighbors each candidate rectangle should have to retain it
        :param single: search for single (True) or multiple (False) faces
        :param debug: if True, enables verbose output
        :param grayscale: if True, converts the image to grayscale
        :return: greyscale region(s) of interest, scaled to 64x64 pixels
        """

        if img is None:
            return

        # Creates a Cascade Classifier and loads the default training data
        haar_xml = "../classifiers/haarcascade_frontalface_default.xml"
        if not os.path.isfile(haar_xml):
            print("[ERROR] Unable to load the HaarCascade classifier. Verify file: \"" + os.path.relpath(haar_xml)
                  + "\"")
            quit(-1)
        face_cascade = cv2.CascadeClassifier(haar_xml)

        if grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Histogram equalization
            gray = cv2.equalizeHist(gray)
        else:
            # I keep the variable name because this is a late update
            gray = img
        if debug:
            # Shows the color (and eventually the gray) image
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            if grayscale:
                cv2.imshow("gray", gray)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        faces = face_cascade.detectMultiScale3(gray, scaleFactor=scale_factor, minNeighbors=min_neighbours,
                                               outputRejectLevels=True)
        rects = faces[0]
        #neighbours = faces[1]
        weights = faces[2]
        roi_list = []
        roi_areas = []
        c = 0
        for (x, y, w, h) in rects:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, str(weights[c][0]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if debug:
                cv2.imshow("rectangled", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            roi = gray[y:y + h, x:x + w]
            roi_area = functools.reduce(lambda x, y: x * y, roi.shape)
            roi_resized = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
            roi_list.append(roi_resized)
            roi_areas.append(roi_area)
            c += 1

        if len(roi_list) == 0:
            return None
        elif single:
            # Gets the biggest rectangle (nearest to the robot)
            max_value = max(roi_areas)
            max_index = roi_areas.index(max_value)
            return roi_list[max_index]
        else:
            return roi_list

    # --- RECOGNITION ---

    # Selects a model
    @staticmethod
    def model_initialize(model_number, withTreshold=False, threshold=100.0):
        if model_number == 0:
            if withTreshold:
                return cv2.face.createEigenFaceRecognizer(threshold=threshold)
            else:
                return cv2.face.createEigenFaceRecognizer()
        elif model_number == 1:
            if withTreshold:
                return cv2.face.createFisherFaceRecognizer(threshold=threshold)
            else:
                return cv2.face.createFisherFaceRecognizer()
        elif model_number == 2:
            if withTreshold:
                return cv2.face.createLBPHFaceRecognizer(threshold=threshold)
            else:
                return cv2.face.createLBPHFaceRecognizer()
        else:
            print("[ERROR] Invalid algorithm selected: " + str(FaceVision.ALGORITHM_NUMBER))
        quit()

    # Acquires training data from a directory containing images
    @staticmethod
    def data_from_file(directory):
        data = TrainingData()
        informers = 2
        for i in range(informers):
            directory = os.path.join(directory, str(i))
            file_list = [f for f in os.listdir(directory)]
            for file in file_list:
                img = cv2.imread(os.path.join(directory, file))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_equ = cv2.equalizeHist(gray)
                data.images.append(gray_equ)
                data.labels.append(i)
        return data

    # Trains the face recognition module using the selected model. Saves is for future use.
    @staticmethod
    def recognition_train(data):
        if isinstance(data, TrainingData):
            model = FaceVision.model_initialize(FaceVision.ALGORITHM_NUMBER, withTreshold=False)
            model.train(data.images, data.labels)
            # Cleares up previous models
            if os.path.exists(FaceVision.MODEL_FILE):
                os.remove(FaceVision.MODEL_FILE)
            model.save(FaceVision.MODEL_FILE)
        else:
            print("[ERROR] recognition_train: input is not a TrainingData instance.")
            quit(-1)

    # Loads the selected model and does a prediction.
    # Threshold regulates the unknown informant detection
    # I assume frame is already been cropped, resized and converted to greyscale
    @staticmethod
    def recognition_predict(frame):
        model = FaceVision.model_initialize(FaceVision.ALGORITHM_NUMBER, withTreshold=True)
        model.load(FaceVision.MODEL_FILE)
        [predicted_label, predicted_confidence] = model.predict(frame)
        # Returns class name
        return predicted_label

    # Updates the model with new training data
    @staticmethod
    def recognition_update(new_data):
        if isinstance(new_data, TrainingData):
            model = FaceVision.model_initialize(FaceVision.ALGORITHM_NUMBER, withTreshold=False)
            model.load(FaceVision.MODEL_FILE)
            model.update(new_data.images, new_data.labels)
            # Cleares up previous models
            if os.path.exists(FaceVision.MODEL_FILE):
                os.remove(FaceVision.MODEL_FILE)
            model.save(FaceVision.MODEL_FILE)
        else:
            print("[ERROR] recognition_update: input is not a TrainingData instance.")
            quit(-1)


"""
Support class to manage the training dataset for the face recognition algorithms.
"""


class TrainingData:
    def __init__(self):
        self.images = []
        self.labels = []

    # Converts images from OpenCV image to nparray and generates a new TrainingData object to be passed
    # to the training function
    def prepare_for_training(self):
        nparrays = []
        for image in self.images:
            nparrays.append(np.asarray(image, dtype=np.uint8))
        new_item = TrainingData()
        new_item.images = nparrays
        new_item.labels = np.asarray(self.labels, dtype=np.int32)
        return new_item
