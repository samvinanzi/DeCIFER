"""
Class for skeletal data processing and management
"""

import PyOpenPose as OP
import cv2
import numpy as np
import os
from Keypoint import Keypoint
import matplotlib.pyplot as plt


OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


class Skeleton:
    def __init__(self, image):
        self.rgb = image
        self.keypoints = {}
        self.prepare()

    # Prepares the data for usage and clustering
    def prepare(self):
        self.get_keypoints()
        #self.convert_to_cartesian()
        #self.cippitelli_norm()

    # Retrieves the skeletal keypoints
    def get_keypoints(self):
        # OpenPose parameters
        net_pose_size = (320, 240)
        net_face_hands_size = (240, 240)
        output_size = (640, 480)
        model = "COCO"
        model_dir = OPENPOSE_ROOT + os.sep + "models" + os.sep
        log = 0
        heatmaps = False
        heatmaps_scale = OP.OpenPose.ScaleMode.PlusMinusOne
        with_face = False
        with_hands = False
        gpu_id = 0
        op = OP.OpenPose(net_pose_size, net_face_hands_size, output_size, model, model_dir, log, heatmaps,
                         heatmaps_scale, with_face, with_hands, gpu_id)
        op.detectPose(self.rgb)
        keypoints = op.getKeypoints(op.KeypointType.POSE)[0]
        if keypoints is None:
            print("No humans found in this frame!")
            quit()
        humans_found = keypoints.shape[0]
        if humans_found > 1:
            print("Warning: more than one human found in this frame.")
        else:
            print("One human detected.")
        keypoints = keypoints[0]
        # Keypoint reduction
        # Calculates a new keypoint for the hips as the median between points 8 and 11 fo the original skeleton
        point8 = keypoints[8]
        point11 = keypoints[11]
        newpoint = np.array([(point8[0] + point11[0]) / 2, (point8[1] + point11[1]) / 2, (point8[2] + point11[2]) / 2])
        # Delete unwanted keypoints
        keypoints = np.delete(keypoints, [2, 5, 8, 11, 14, 15, 16, 17], 0)
        # Add the new keypoint
        keypoints = np.vstack([keypoints, newpoint])
        # Saves them as a dictionary of Keypoints objects
        self.keypoints = {
            "Head": Keypoint(keypoints[0][0], keypoints[0][1], keypoints[0][2]),
            "Neck": Keypoint(keypoints[1][0], keypoints[1][1], keypoints[1][2]),
            "RElbow": Keypoint(keypoints[2][0], keypoints[2][1], keypoints[2][2]),
            "RWrist": Keypoint(keypoints[3][0], keypoints[3][1], keypoints[3][2]),
            "LElbow": Keypoint(keypoints[4][0], keypoints[4][1], keypoints[4][2]),
            "LWrist": Keypoint(keypoints[5][0], keypoints[5][1], keypoints[5][2]),
            "RKnee": Keypoint(keypoints[6][0], keypoints[6][1], keypoints[6][2]),
            "RAnkle": Keypoint(keypoints[7][0], keypoints[7][1], keypoints[7][2]),
            "LKnee": Keypoint(keypoints[8][0], keypoints[8][1], keypoints[8][2]),
            "LAnkle": Keypoint(keypoints[9][0], keypoints[9][1], keypoints[9][2]),
            "Torso": Keypoint(keypoints[10][0], keypoints[10][1], keypoints[10][2])
        }

    # Computes the missing keypoint names
    def get_missing_keypoints(self):
        output = []
        for name, keypoint in self.keypoints.items():
            if keypoint.is_empty():
                output.append(name)
        return output

    # Returns the non-missing keypoint dictionary
    def nonmissing_keypoins(self):
        missing_keypoints = self.get_missing_keypoints()
        return {key: self.keypoints[key] for key in self.keypoints if key not in missing_keypoints}

    # Converts the dictionary of Keypoints into a numpy array
    # It is unordered, but it doesn't matter
    def keypoints_to_array(self, keypoints=None):
        if keypoints is None:
            keypoints = self.keypoints
        output = []
        for _, keypoint in keypoints.items():
            output.append([keypoint.x, keypoint.y])
        return np.array(output)

    # Computes the minimum bounding box around the detected skeletal keypoints
    def bounding_box(self, padding=0.2):
        # Remove from the computation the missing keypoints
        temp_keypoints = self.nonmissing_keypoins()
        # Converts the new keypoints to numpy array
        temp_keypoints = self.keypoints_to_array(temp_keypoints)

        minX = np.min(temp_keypoints[:, 0])
        minY = np.min(temp_keypoints[:, 1])

        maxX = np.max(temp_keypoints[:, 0])
        maxY = np.max(temp_keypoints[:, 1])

        width = maxX - minX
        height = maxY - minY

        padX = width * padding / 2
        padY = height * padding / 2

        minX -= padX
        minY -= padY

        width += 2 * padX
        height += 2 * padY

        return [int(minX), int(minY), int(minX) + int(width), int(minY) + int(height)]

    # Crops and displays the skeleton, with or without the background generating image
    def show(self, color=(255, 255, 255), background=False, save=False, savename="skeleton"):
        # Sets the background
        if background:
            image = self.rgb
        else:
            height, width, channels = self.rgb.shape
            image = np.zeros((height, width, channels), np.uint8)
        box = self.bounding_box()     # Fetches the bounding box dimensions
        # Iterates for each non-missing point
        for name, keypoint in self.nonmissing_keypoins().items():
            cv2.circle(image, (int(keypoint.x), int(keypoint.y)), 3, color, 5)
            cv2.putText(image, name, (int(keypoint.x), int(keypoint.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        # Crops the image
        roi = image[box[1]:box[3], box[0]:box[2]]
        if save:
            cv2.imwrite("img/test/" + savename + ".jpg", roi)
        cv2.imshow("Skeletal ROI", roi)
        cv2.waitKey(0)

    # Cippitelli normalization
    def cippitelli_norm(self):
        J0 = self.keypoints['Torso']
        J2 = self.keypoints['Neck']
        distance = J2.distance_to(J0)
        for name, Ji in self.nonmissing_keypoins().items():
            di = Keypoint()
            di.x = (Ji.x - J0.x) / distance
            di.y = (Ji.y - J0.y) / distance
            # Substitute this point to the original one
            self.keypoints[name] = di

    # Plots the data points
    def plot(self, save=False):
        array = self.keypoints_to_array()
        x = array[:, 0]
        y = array[:, 1]
        plt.plot(x, y, 'bo')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Skeletal Keypoints')
        plt.grid(True)
        if save:
            plt.savefig("plot.png")
        plt.show()

    # Converts pixel coordinates to cartesian
    def convert_to_cartesian(self):
        # Work on a copy of the original data
        kps = self.nonmissing_keypoins()
        height, width, _ = self.rgb.shape
        for name, kp in kps.items():
            kp.x = kp.x - width / 2
            kp.y = -kp.y + height / 2
            self.keypoints[name] = kp

    # Converts cartesian coordinates to pixel
    def convert_to_pixel(self):
        # Work on a copy of the original data
        kps = self.nonmissing_keypoins()
        height, width, _ = self.rgb.shape
        for name, kp in kps.items():
            kp.x = kp.x + width/2
            kp.y = -kp.y + height/2
            self.keypoints[name] = kp
