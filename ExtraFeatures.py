"""

Feature engineering for finer clustering of human skeletons.

"""

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
import time
import matplotlib.pyplot as plt
import math


class ExtraFeatures:
    def __init__(self, skeleton):
        self.skeleton = skeleton
        self.features = []
        # Computation of several features. Comment out the undesired ones.
        self.elbow_angle()
        #self.gaze_direction()

    # Returns the full new feature list
    def get_features(self):
        return np.array(self.features)

    # Calculates the angles in degrees formed by the arm and forearm of both sides.
    # Adds two features: [right_angle, left_angle]
    '''
                   a    <- shoulder
                  /
                 /
    elbow ->    b -----c    <- wrist
    '''
    def elbow_angle(self, debug=False):
        angles = []
        keypoints = {
            "right": ["RShoulder", "RElbow", "RWrist"],
            "left": ["LShoulder", "LElbow", "LWrist"]
        }
        if debug:
            ax = plt.gca()
        img = self.skeleton.img
        for side, joints in iter(sorted(keypoints.items(), reverse=True)):
            try:
                a = np.array(self.skeleton.keypoints[joints[0]].as_list())
                b = np.array(self.skeleton.keypoints[joints[1]].as_list())
                c = np.array(self.skeleton.keypoints[joints[2]].as_list())
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                #angles.append(np.degrees(angle))   # Angle in degrees
                angles.append(math.pi - angle)      # Delta angle in radians
                # Debug output
                if debug:
                    print(side + ": " + str(np.degrees(angle)))
                    print(side + " (radians): " + str(angle))
                    ax.plot(np.linspace(a[0], b[0]), np.linspace(a[1], b[1]), c=("blue" if side == "right" else "red"), marker='.', linestyle=':', linewidth=0.1)
                    ax.plot(np.linspace(b[0], c[0]), np.linspace(b[1], c[1]), c=("blue" if side == "right" else "red"), marker='.', linestyle=':', linewidth=0.1)
            except KeyError:
                print("[WARNING] Some arm keypoints have not been found. Elbow angles not computable.")
                angles.append(0.0)
        if debug:
            plt.show()
        self.features.extend(angles)

    # Estimate the gaze direction
    # Adds up to three features: [roll, pitch, yaw]
    def gaze_direction(self, roll=False, pitch=True, yaw=True, verbose=True):
        assert roll or pitch or yaw, "At least one of the dimensions must be true."
        # Initialization todo farlo una volta sola come singleton?
        # --- #
        sess = tf.Session()  # Launch the graph in a session.
        my_head_pose_estimator = CnnHeadPoseEstimator(sess)  # Head pose estimation object
        # Load the weights from the configuration folders
        tic = time.time()
        weights_path = "/home/samuele/Research/repositories/deepgaze/etc/tensorflow/head_pose/{}/cnn_cccdd_30k.tf"
        my_head_pose_estimator.load_roll_variables(os.path.realpath(weights_path.format("roll")))
        my_head_pose_estimator.load_pitch_variables(os.path.realpath(weights_path.format("pitch")))
        my_head_pose_estimator.load_yaw_variables(os.path.realpath(weights_path.format("yaw")))
        print("LOAD: " + str(time.time() - tic))
        # --- #
        image = self.skeleton.img   # Selects the cropped skeleton image
        if verbose:
            print("[DEBUG] Predicting gaze for skeleton " + str(self.skeleton.id))
        # Resize the image to be a square encompassing the upper part of the picture
        h, w, d = image.shape
        image = image[0:w, :, :]
        # Calculate R/P/Y
        tic = time.time()
        if roll:
            roll = my_head_pose_estimator.return_roll(image)  # Evaluate the roll angle using a CNN
            self.features.extend([roll[0, 0, 0]])
        if pitch:
            pitch = my_head_pose_estimator.return_pitch(image)  # Evaluate the pitch angle using a CNN
            self.features.extend([pitch[0, 0, 0]])
        if yaw:
            yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
            self.features.extend([yaw[0, 0, 0]])
        print("PREDICT: " + str(time.time() - tic))
        sess.close()

