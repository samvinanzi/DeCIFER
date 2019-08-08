"""

Feature engineering for finer clustering of human skeletons.

"""

import numpy as np


class ExtraFeatures:
    def __init__(self, keypoints):
        self.kp = keypoints
        self.features = []
        # Computation of several features. Comment out the undesired ones.
        self.elbow_angle()

    # Returns the full new feature list
    def get_features(self):
        return np.array(self.features)

    # Calculates the angles in degrees formed by the arm and forearm of both sides, returns them in a list (left, right)
    '''
                   a    <- shoulder
                  /
                 /
    elbow ->    b -----c    <- wrist
    '''
    def elbow_angle(self):
        angles = []
        keypoints = {
            "right": ["RShoulder", "RElbow", "RWrist"],
            "left": ["LShoulder", "LElbow", "LWrist"]
        }
        for _, joints in iter(sorted(keypoints.items())):
            try:
                a = np.array(self.kp[joints[0]].as_list())
                b = np.array(self.kp[joints[1]].as_list())
                c = np.array(self.kp[joints[2]].as_list())
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                angles.append(np.degrees(angle))
            except KeyError:
                print("[WARNING] Some arm keypoints have not been found. Elbow angles not computable.")
                angles.append(0.0)
        self.features.extend(angles)
