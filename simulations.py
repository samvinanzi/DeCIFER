"""
Simulations scripts are stored here
"""

from Skeleton import Skeleton
import matplotlib.pyplot as plt
import cv2


def show_normalization_effect():
    image = cv2.imread("human.jpg")
    skeleton = Skeleton(image)
    skeleton.convert_to_cartesian()

    a = skeleton.keypoints_to_array()

    skeleton.cippitelli_norm()

    b = skeleton.keypoints_to_array()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    f.suptitle('Cippitelli Normalization effect on skeletal data')
    ax1.plot_keypoints(a[:, 0], a[:, 1], 'bo')
    ax2.plot_keypoints(b[:, 0], b[:, 1], 'bo')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title("Pre-normalization")
    ax2.set_title("Post-normalization")
    plt.savefig("img/simulations/cippitelli_norm_effect.png")
    plt.show()