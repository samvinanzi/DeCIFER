"""

A wrapper class that initializes and manages the deepgaze module for gaze prediction. It also instantiates a single
instance of this class to load the model weights from disk only once.

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # Avoids warnings clogging the output
import tensorflow as tf
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator


class DeepGazeWrapper:
    def __init__(self, radians=True):
        self.sess = tf.Session()  # Launch the graph in a session.
        self.my_head_pose_estimator = CnnHeadPoseEstimator(self.sess)  # Head pose estimation object
        weights_path = "/home/samuele/Research/repositories/deepgaze/etc/tensorflow/head_pose/{}/cnn_cccdd_30k.tf"
        self.my_head_pose_estimator.load_roll_variables(os.path.realpath(weights_path.format("roll")))
        self.my_head_pose_estimator.load_pitch_variables(os.path.realpath(weights_path.format("pitch")))
        self.my_head_pose_estimator.load_yaw_variables(os.path.realpath(weights_path.format("yaw")))
        self.radians = radians
        # Reference frame for the values, in radians, of r/p/y
        # These values were taken from a frame showing the human looking directly at the robot camera.
        self.reference = {
            'roll': -0.01998772,
            'pitch': -0.07902261,
            'yaw': -0.75663966
        }

    # Resize the image to be a square encompassing the upper part of the picture
    def resize_input(self, image):
        h, w, d = image.shape
        return image[0:w, :, :]

    def get_roll(self, image):
        return self.my_head_pose_estimator.return_roll(image, radians=self.radians) - self.reference['roll']

    def get_pitch(self, image):
        return self.my_head_pose_estimator.return_pitch(image, radians=self.radians) - self.reference['pitch']

    def get_yaw(self, image):
        return self.my_head_pose_estimator.return_yaw(image, radians=self.radians) - self.reference['yaw']

    def close(self):
        self.sess.close()


dg = DeepGazeWrapper()
