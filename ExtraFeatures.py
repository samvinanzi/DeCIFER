"""

Feature engineering for finer clustering of human skeletons.

"""

import numpy as np
import matplotlib.pyplot as plt
import math
from DeepGazeWrapper import dg as deepgaze


class ExtraFeatures:
    def __init__(self, skeleton, angles=False, gaze=True):
        self.skeleton = skeleton
        self.features = []
        if angles:
            self.elbow_angle()
        if gaze:
            self.gaze_direction()

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
                    ax.plot(np.linspace(a[0], b[0]), np.linspace(a[1], b[1]), c=("blue" if side == "right" else "red"),
                            marker='.', linestyle=':', linewidth=0.1)
                    ax.plot(np.linspace(b[0], c[0]), np.linspace(b[1], c[1]), c=("blue" if side == "right" else "red"),
                            marker='.', linestyle=':', linewidth=0.1)
            except KeyError:
                print("[WARNING] Some arm keypoints have not been found. Elbow angles not computable.")
                angles.append(0.0)
        if debug:
            plt.show()
        self.features.extend(angles)

    # Estimate the gaze direction.
    # Adds up to three features: [roll, pitch, yaw]
    def gaze_direction(self, roll=True, pitch=True, yaw=True, debug=True):
        assert roll or pitch or yaw, "At least one of the dimensions must be true."
        # Resize the image to be a square encompassing the upper part of the picture
        image = deepgaze.resize_input(self.skeleton.img)
        if debug:
            print("[DEBUG] Predicting gaze for skeleton " + str(self.skeleton.id))
        # Calculate R/P/Y
        if roll:
            roll = deepgaze.get_roll(image)  # Evaluate the roll angle using a CNN
            self.features.extend([roll[0, 0, 0]])
            if debug:
                print("ROLL: " + str(roll))
        if pitch:
            pitch = deepgaze.get_pitch(image)  # Evaluate the pitch angle using a CNN
            self.features.extend([pitch[0, 0, 0]])
            if debug:
                print("PITCH: " + str(pitch))
        if yaw:
            yaw = deepgaze.get_yaw(image)  # Evaluate the yaw angle using a CNN
            self.features.extend([yaw[0, 0, 0]])
            if debug:
                print("YAW: " + str(yaw))
