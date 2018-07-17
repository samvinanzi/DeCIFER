"""

Class for skeletal data processing and management. Extracts skeletons from RGB images.

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
    def __init__(self, image, robot, id=0):
        self.origin = image.copy()
        self.keypoints = {}
        self.keypoints_2d = {}
        self.img = None
        self.id = id
        # Performs computations
        self.prepare(robot)

    # Prepares the data for usage and clustering
    def prepare(self, robot):
        self.get_keypoints(robot)
        # self.convert_to_cartesian() # Not needed, SFM already converts pixel to cartesian
        self.cippitelli_norm()

    # Retrieves the skeletal keypoints
    def get_keypoints(self, robot):
        op.detectPose(self.origin)
        keypoints = op.getKeypoints(op.KeypointType.POSE)[0]
        if keypoints is None:
            print("No humans found in this frame!")
            raise NoHumansFoundException
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
        # Removes the "confidence" column
        keypoints = np.delete(keypoints, 2, axis=1)
        # Converts the keypoints to 3D representation
        keypoints3d = robot.request_3d_points(keypoints.tolist())
        #keypoints3d = np.append(keypoints, np.zeros((11, 1)), axis=1)   # 2D degeneration, for testing
        # Saves them as a dictionary of Keypoints objects
        self.keypoints = {
            "Head": Keypoint(keypoints3d[0][0], keypoints3d[0][1], keypoints3d[0][2]),
            "Neck": Keypoint(keypoints3d[1][0], keypoints3d[1][1], keypoints3d[1][2]),
            "RElbow": Keypoint(keypoints3d[2][0], keypoints3d[2][1], keypoints3d[2][2]),
            "RWrist": Keypoint(keypoints3d[3][0], keypoints3d[3][1], keypoints3d[3][2]),
            "LElbow": Keypoint(keypoints3d[4][0], keypoints3d[4][1], keypoints3d[4][2]),
            "LWrist": Keypoint(keypoints3d[5][0], keypoints3d[5][1], keypoints3d[5][2]),
            "RKnee": Keypoint(keypoints3d[6][0], keypoints3d[6][1], keypoints3d[6][2]),
            "RAnkle": Keypoint(keypoints3d[7][0], keypoints3d[7][1], keypoints3d[7][2]),
            "LKnee": Keypoint(keypoints3d[8][0], keypoints3d[8][1], keypoints3d[8][2]),
            "LAnkle": Keypoint(keypoints3d[9][0], keypoints3d[9][1], keypoints3d[9][2]),
            "Torso": Keypoint(keypoints3d[10][0], keypoints3d[10][1], keypoints3d[10][2])
        }
        # These operations are postponed to avoid time delays with the 3D coordinate generation
        # Saves the 2D pixel representation of the keypoints
        self.keypoints_2d = {
            "Head": Keypoint(keypoints[0][0], keypoints[0][1]),
            "Neck": Keypoint(keypoints[1][0], keypoints[1][1]),
            "RElbow": Keypoint(keypoints[2][0], keypoints[2][1]),
            "RWrist": Keypoint(keypoints[3][0], keypoints[3][1]),
            "LElbow": Keypoint(keypoints[4][0], keypoints[4][1]),
            "LWrist": Keypoint(keypoints[5][0], keypoints[5][1]),
            "RKnee": Keypoint(keypoints[6][0], keypoints[6][1]),
            "RAnkle": Keypoint(keypoints[7][0], keypoints[7][1]),
            "LKnee": Keypoint(keypoints[8][0], keypoints[8][1]),
            "LAnkle": Keypoint(keypoints[9][0], keypoints[9][1]),
            "Torso": Keypoint(keypoints[10][0], keypoints[10][1])
        }
        # Generates the image based on the 2D pixel keypoint
        self.generate_image()

    # Computes the missing keypoint names
    def get_missing_keypoints(self, apply_on_2d=False):
        if apply_on_2d:
            items = self.keypoints_2d.items()
        else:
            items = self.keypoints.items()
        output = []
        for name, keypoint in items:
            if keypoint.is_empty():
                output.append(name)
        return output

    # Returns the non-missing keypoint dictionary
    def nonmissing_keypoints(self, apply_to_2d=False):
        if apply_to_2d:
            keypoints = self.keypoints_2d
        else:
            keypoints = self.keypoints
        missing_keypoints = self.get_missing_keypoints(apply_to_2d)
        return {key: keypoints[key] for key in keypoints if key not in missing_keypoints}

    # Converts the dictionary of Keypoints into an ordered numpy array
    # To use it on the 2D keypoint set, just pass it as an argument
    def keypoints_to_array(self, keypoints=None):
        # This condition enables the computation even on other, external or temp keypoint arrays
        if keypoints is None:
            keypoints = self.keypoints
        # Compuation starts here
        output = []
        # Orders the joints alphabetically
        joints = sorted(keypoints)
        for joint in joints:
            output.append([keypoints[joint].x, keypoints[joint].y, keypoints[joint].z])
        return np.array(output)

    # Computes the minimum 2D bounding box around the detected skeletal keypoints
    # Works on the 2D keypoints dataset
    def bounding_box(self, padding=0.2):
        # Remove from the computation the missing keypoints
        temp_keypoints = self.nonmissing_keypoints(apply_to_2d=True)
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

    # Generates a B/W image of the skeleton and stores it
    def generate_image(self):
        # Sets the background
        self.img = self.origin.copy()
        box = self.bounding_box()  # Fetches the bounding box dimensions
        # Iterates for each non-missing point
        for name, keypoint in self.nonmissing_keypoints(apply_to_2d=True).items():
            cv2.circle(self.img, (int(keypoint.x), int(keypoint.y)), 1, (0, 255, 255), 5)
            # cv2.putText(image, name, (int(keypoint.x), int(keypoint.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
        # Crops the image
        self.img = self.img[box[1]:box[3], box[0]:box[2]].copy()

    # Cippitelli normalization
    def cippitelli_norm(self):
        torso = self.keypoints['Torso']
        neck = self.keypoints['Neck']
        distance = neck.distance_to(torso)
        for name, Ji in self.nonmissing_keypoints().items():
            di = Keypoint()
            di.x = (Ji.x - torso.x) / distance
            di.y = (Ji.y - torso.y) / distance
            di.z = (Ji.z - torso.z) / distance
            # Substitute this point to the original one
            self.keypoints[name] = di

    # Returns a row array (1x30) of features for this skeleton as a dataset example
    def as_feature(self):
        array = self.keypoints_to_array()
        # Deletes the final row corresponding to Torso.x and Torso.y (they are always zero)
        array = array[:-1, :]
        return np.array(array).ravel()

    # Converts pixel coordinates to cartesian
    def convert_to_cartesian(self):
        # Work on a copy of the original data
        kps = self.nonmissing_keypoints(apply_to_2d=True)
        height, width, _ = self.origin.shape
        for name, kp in kps.items():
            kp.x = kp.x - width / 2
            kp.y = -kp.y + height / 2
            self.keypoints[name] = kp

    # Converts cartesian coordinates to pixel
    def convert_to_pixel(self):
        # Work on a copy of the original data
        kps = self.nonmissing_keypoints(apply_to_2d=True)
        height, width, _ = self.origin.shape
        for name, kp in kps.items():
            kp.x = kp.x + width/2
            kp.y = -kp.y + height/2
            self.keypoints[name] = kp

    # Analyzes the keypoints to determine the orientation of the person
    def orientation_reach(self, factor=2.0):
        # Uses the 4 arms keypoints, if they are valid
        keypoints = ['RWrist', 'LWrist', 'RElbow', 'LElbow']
        arms_points = {}
        for keypoint in keypoints:
            if not self.keypoints_2d[keypoint].is_empty():
                arms_points[keypoint] = self.keypoints_2d[keypoint]
        array_keypoints = self.keypoints_to_array(arms_points)
        # Fetches the leftmost and rightmost points on the horizontal axis
        leftmost = np.min(array_keypoints[:, 0])
        rightmost = np.max(array_keypoints[:, 0])
        # Uses the neck as a reference
        neck_x = self.keypoints_2d['Neck'].x
        # Calculates the two distances
        d_left = neck_x - leftmost
        d_right = rightmost - neck_x
        # Based on the distance, determines where the person is reaching
        if d_left > factor * d_right:
            return "left"
        elif d_right > factor * d_left:
            return "right"
        else:
            return "center"

    # ---- DISPLAY FUNCTIONS ----

    # Plots the data points
    def plot_keypoints(self, save=False, apply_to_2d=False):
        if apply_to_2d:
            print("[WARNING] 2D keypoints have not been converted to cartesian space.")
            array = self.keypoints_to_array(self.keypoints_2d)
        else:
            array = self.keypoints_to_array()
        x = array[:, 0]
        y = array[:, 1]
        z = array[:, 2]
        ax = plt.axes(projection='3d')
        ax.scatter(x, y, z, c='b', marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Skeletal Keypoints')
        plt.grid(True)
        # Puts text
        for label, keypoint in self.keypoints.items():
            if not keypoint.is_empty():
                ax.text(keypoint.x, keypoint.y, keypoint.z, label, None)
        if save:
            plt.savefig("plot.png")
        plt.show()

    # Quickly displays the associated image for the skeleton.
    def display_fast(self):
        cv2.imshow("im", self.img)
        cv2.waitKey(0)

    # Image processing and visualization. Crops and displays the skeleton, with or without the background image
    def display(self, color=(255, 255, 255), background=False, save=False, savename="skeleton"):
        # Sets the background
        if background:
            image = self.origin.copy()
        else:
            height, width, channels = self.origin.shape
            image = np.zeros((height, width, channels), np.uint8)
        box = self.bounding_box()  # Fetches the bounding box dimensions
        # Iterates for each non-missing point
        for name, keypoint in self.nonmissing_keypoints(apply_to_2d=True).items():
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


# Custom exception
class NoHumansFoundException(Exception):
    pass
