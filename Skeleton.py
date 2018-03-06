"""
Class for skeletal data processing and management
"""

import PyOpenPose as PyOP
import cv2
import numpy as np
import os
from Keypoint import Keypoint
import matplotlib.pyplot as plt

# ----- BEGIN PyOpenPose initialization ----- #

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]

net_pose_size = (320, 240)
net_face_hands_size = (240, 240)
output_size = (640, 480)
model = "COCO"
model_dir = OPENPOSE_ROOT + os.sep + "models" + os.sep
log = 0
heatmaps = False
heatmaps_scale = PyOP.OpenPose.ScaleMode.PlusMinusOne
with_face = False
with_hands = False
gpu_id = 0
op = PyOP.OpenPose(net_pose_size, net_face_hands_size, output_size, model, model_dir, log, heatmaps,
                   heatmaps_scale, with_face, with_hands, gpu_id)

# ----- END PyOpenPose initialization ----- #


class Skeleton:
    def __init__(self, image, id=0):
        # Member initialization
        self.origin = image.copy()
        self.keypoints = {}
        self.img = None
        self.id = id
        # Performs computations
        self.prepare()

    # Prepares the data for usage and clustering
    def prepare(self):
        self.get_keypoints()
        self.generate_image()
        self.convert_to_cartesian()
        self.cippitelli_norm()

    # Retrieves the skeletal keypoints
    def get_keypoints(self):
        op.detectPose(self.origin)
        keypoints = op.getKeypoints(op.KeypointType.POSE)[0]
        if keypoints is None:
            print("No humans found in this frame!")
            quit()
        humans_found = keypoints.shape[0]
        if humans_found > 1:
            print("[" + str(self.id) + "] Warning: more than one human found in this frame.")
        else:
            print("[" + str(self.id) + "] One human detected.")
        keypoints = keypoints[0]
        # KEYPOINT REDUCTION
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
    def nonmissing_keypoints(self):
        missing_keypoints = self.get_missing_keypoints()
        return {key: self.keypoints[key] for key in self.keypoints if key not in missing_keypoints}

    # Converts the dictionary of Keypoints into an ordered numpy array
    def keypoints_to_array(self, keypoints=None):
        # This condition enables the computation even on other, external or temp keypoint arrays
        if keypoints is None:
            keypoints = self.keypoints
        # Compuation starts here
        output = []
        # Orders the joints alphabetically
        joints = sorted(keypoints)
        for joint in joints:
            output.append([keypoints[joint].x, keypoints[joint].y])
        return np.array(output)

    # Computes the minimum bounding box around the detected skeletal keypoints
    def bounding_box(self, padding=0.2):
        # Remove from the computation the missing keypoints
        temp_keypoints = self.nonmissing_keypoints()
        # Converts the new keypoints to numpy array
        temp_keypoints = self.keypoints_to_array(temp_keypoints)
        # Min
        min_x = np.min(temp_keypoints[:, 0])
        min_y = np.min(temp_keypoints[:, 1])
        # Max
        max_x = np.max(temp_keypoints[:, 0])
        max_y = np.max(temp_keypoints[:, 1])
        # Dimensions
        width = max_x - min_x
        height = max_y - min_y
        # Padding
        pad_x = width * padding / 2
        pad_y = height * padding / 2
        # Effective measures
        min_x -= pad_x
        min_y -= pad_y
        width += 2 * pad_x
        height += 2 * pad_y
        # Sanity check to avoid negative coordinates
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        return [int(min_x), int(min_y), int(min_x) + int(width), int(min_y) + int(height)]

    # Crops and displays the skeleton, with or without the background generating image
    def show(self, color=(255, 255, 255), background=False, save=False, savename="skeleton"):
        if self.img is not None:
            print("Warning! show() cannot be used for transformed and normalized images. Use display() instead.")
            return None
        # Sets the background
        if background:
            image = self.origin.copy()
        else:
            height, width, channels = self.origin.shape
            image = np.zeros((height, width, channels), np.uint8)
        box = self.bounding_box()     # Fetches the bounding box dimensions
        # Iterates for each non-missing point
        for name, keypoint in self.nonmissing_keypoints().items():
            cv2.circle(image, (int(keypoint.x), int(keypoint.y)), 3, color, 5)
            cv2.putText(image, name, (int(keypoint.x), int(keypoint.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        # Crops the image
        roi = image[box[1]:box[3], box[0]:box[2]]
        if save:
            cv2.imwrite("img/test/" + savename + ".jpg", roi)
        cv2.imshow("Skeletal ROI", roi)
        cv2.waitKey(0)
        # Returns the image, in case some other component needs it
        return roi

    # Generates a B/W image and stores it
    def generate_image(self):
        # Sets the background
        #height, width, channels = self.origin.shape
        #self.img = np.zeros((height, width, channels), np.uint8)
        self.img = self.origin.copy()
        box = self.bounding_box()  # Fetches the bounding box dimensions
        # Iterates for each non-missing point
        for name, keypoint in self.nonmissing_keypoints().items():
            cv2.circle(self.img, (int(keypoint.x), int(keypoint.y)), 1, (0, 255, 255), 5)
            # cv2.putText(image, name, (int(keypoint.x), int(keypoint.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        # Crops the image
        self.img = self.img[box[1]:box[3], box[0]:box[2]].copy()

    # Visualizes the skeleton image. To be used when the keypoints have already been normalized
    def display(self):
        cv2.imshow("im", self.img)
        cv2.waitKey(0)

    # Cippitelli normalization
    def cippitelli_norm(self):
        torso = self.keypoints['Torso']
        neck = self.keypoints['Neck']
        distance = neck.distance_to(torso)
        for name, Ji in self.nonmissing_keypoints().items():
            di = Keypoint()
            di.x = (Ji.x - torso.x) / distance
            di.y = (Ji.y - torso.y) / distance
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
        kps = self.nonmissing_keypoints()
        height, width, _ = self.origin.shape
        for name, kp in kps.items():
            kp.x = kp.x - width / 2
            kp.y = -kp.y + height / 2
            self.keypoints[name] = kp

    # Converts cartesian coordinates to pixel
    def convert_to_pixel(self):
        # Work on a copy of the original data
        kps = self.nonmissing_keypoints()
        height, width, _ = self.origin.shape
        for name, kp in kps.items():
            kp.x = kp.x + width/2
            kp.y = -kp.y + height/2
            self.keypoints[name] = kp

    # Returns a row array (1x20) of features for this skeleton as a dataset example
    def as_feature(self):
        array = self.keypoints_to_array()
        # Deletes the final row corresponding to Torso.x and Torso.y (they are always zero)
        array = array[:-1, :]
        return np.array(array).ravel()
