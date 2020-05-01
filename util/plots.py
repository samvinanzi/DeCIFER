"""

A collection of plotting scripts to fill in the results of papers.

"""

from CognitiveArchitecture import CognitiveArchitecture
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from Skeleton import Skeleton
import cv2


# Compares two skeletons: one normalized and one not
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


# Creates L1 clustering plot
def make_l1_plot():
    import math
    import matplotlib.pyplot as plt

    def find_closest_centroid(sample, centroids):
        min_distance = float("inf")
        closest_cluster = None
        for centroid in centroids:
            dist = math.hypot(sample[0] - centroid[0], sample[1] - centroid[1])
            if dist < min_distance:
                min_distance = dist
                closest_cluster = centroid
        return centroids.index(closest_cluster) # returns 0 or 1

    def plot(data_a, data_b, data_c, color1='red', color2='#4267F1', color3='green', marker1='h', marker2="P", marker3="^"):
        data_a = np.array(data_a)
        x_a, y_a = data_a.T
        plt.scatter(x_a, y_a, s=100, color=color1, edgecolors='black', marker=marker1)
        data_b = np.array(data_b)
        x_b, y_b = data_b.T
        plt.scatter(x_b, y_b, s=100, color=color2, edgecolors='black', marker=marker2)
        data_c = np.array(data_c)
        x_c, y_c = data_c.T
        plt.scatter(x_c, y_c, s=100, color=color3, edgecolors='black', marker=marker3)
        plt.show()

    cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
    cog.train(reload=True)
    centroids = [cog.lowlevel.train.clusters[0].centroid, cog.lowlevel.train.clusters[1].centroid,
                 cog.lowlevel.train.clusters[2].centroid]
    data = cog.lowlevel.train.dataset2d
    # Split
    data0 = [x for x in data if find_closest_centroid(x, centroids) == 0]
    data1 = [x for x in data if find_closest_centroid(x, centroids) == 1]
    data2 = [x for x in data if find_closest_centroid(x, centroids) == 2]
    plot(data0, data1, data2)


# Creates L2 clustering plots
def make_l2_plots():
    import math
    import matplotlib.pyplot as plt

    def find_closest_centroid(sample, centroids):
        min_distance = float("inf")
        closest_cluster = None
        for centroid in centroids:
            dist = math.hypot(sample[0] - centroid[0], sample[1] - centroid[1])
            if dist < min_distance:
                min_distance = dist
                closest_cluster = centroid
        return centroids.index(closest_cluster) # returns 0 or 1

    def plot(data_a, data_b, color1='red', color2='blue', marker1='D', marker2='X'):
        data_a = np.array(data_a)
        x_a, y_a = data_a.T
        plt.scatter(x_a, y_a, s=100, color=color1, edgecolors='black', marker=marker1)
        data_b = np.array(data_b)
        x_b, y_b = data_b.T
        plt.scatter(x_b, y_b, s=100, color=color2, edgecolors='black', marker=marker2)
        plt.show()

    cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
    cog.train(reload=True)
    # Cluster correction
    cog.lowlevel.train.clusters[5].centroid = [1.4346541941018343, -0.1571525604543636]
    cog.lowlevel.train.clusters[6].centroid = [1.0692279685864146, -0.02116144880533045]
    # Data gathering
    data1 = cog.lowlevel.train.l2nodes[0].dataset2d
    data2 = cog.lowlevel.train.l2nodes[1].dataset2d
    # Centroids
    centroids1 = (cog.lowlevel.train.clusters[3].centroid, cog.lowlevel.train.clusters[4].centroid)
    centroids2 = (cog.lowlevel.train.clusters[5].centroid, cog.lowlevel.train.clusters[6].centroid)
    # Delete points
    data1.remove([0.08785277626189862, 0.026598793431451814])
    data1.remove([0.16047217639513775, -0.01272962336737077])
    data2.remove([0.7916992841625878, -0.11955464925538437])
    # Divide each data in two groups
    data1_a = [x for x in data1 if x[0] < 0.0]
    data1_b = [x for x in data1 if x[0] >= 0.0]
    data2_a = [x for x in data2 if x[0] < 0.0]
    data2_b = [x for x in data2 if x[0] >= 0.0]
    # Plot
    plot(data1_a, data1_b, '#FF007F', '#00FFFF')
    plot(data2_a, data2_b, 'yellow', 'magenta')

# Single block confusion matrix (values inputted manually)
def single_block_confusion_matrix():
    def compose(s1, n1, s2, n2):
        true = [s1 for i in range(80)]
        predicted = [s1 for i in range(n1)]
        predicted.extend(s2 for i in range(n2))
        return true, predicted

    def build():
        x = []
        y = []
        (true, predicted) = compose('B', 74, 'O', 6)
        x.extend(true)
        y.extend(predicted)
        (true, predicted) = compose('O', 66, 'B', 14)
        x.extend(true)
        y.extend(predicted)
        (true, predicted) = compose('R', 73, 'G', 7)
        x.extend(true)
        y.extend(predicted)
        (true, predicted) = compose('G', 75, 'R', 5)
        x.extend(true)
        y.extend(predicted)
        return x, y

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    labels = ['B', 'O', 'R', 'G']
    x, y = build()
    cm = confusion_matrix(x, y, labels)
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    # Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp = disp.plot(cmap = plt.cm.OrRd)
    plt.title('Single block confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


# Whole goal confusion matrix (values inputted manually)
def multi_block_confusion_matrix():
    def extend(list, symbol, n):
        list.extend([symbol for i in range(n)])

    labels = ["BGOR", "BROG", "GBRO", "GORB", "OGBR", "ORBG", "RBGO", "ROGB"]
    x = []
    y = []
    # BGOR
    # True
    extend(x, "BGOR", 20)
    # Predictions
    extend(y, "BGOR", 16)
    extend(y, "BROG", 2)
    extend(y, "OGBR", 2)

    # BROG
    # True
    extend(x, "BROG", 20)
    # Predictions
    extend(y, "BGOR", 5)
    extend(y, "BROG", 14)
    extend(y, "ORBG", 1)

    # GBRO
    # True
    extend(x, "GBRO", 20)
    # Predictions
    extend(y, "GBRO", 18)
    extend(y, "GORB", 1)
    extend(y, "RBGO", 1)

    # GORB
    # True
    extend(x, "GORB", 20)
    # Predictions
    extend(y, "GBRO", 1)
    extend(y, "GORB", 18)
    extend(y, "ROGB", 1)

    # OGBR
    # True
    extend(x, "OGBR", 20)
    # Predictions
    extend(y, "BGOR", 4)
    extend(y, "OGBR", 15)
    extend(y, "ORBG", 1)

    # ORBG
    # True
    extend(x, "ORBG", 20)
    # Predictions
    extend(y, "BROG", 3)
    extend(y, "ORBG", 17)

    # RBGO
    # True
    extend(x, "RBGO", 20)
    # Predictions
    extend(y, "GBRO", 2)
    extend(y, "RBGO", 16)
    extend(y, "ROGB", 2)

    # ROGB
    # True
    extend(x, "ROGB", 20)
    # Predictions
    extend(y, "RBGO", 6)
    extend(y, "ROGB", 14)

    cm = confusion_matrix(x, y, labels)
    # Normalize
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)
    # Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp = disp.plot(cmap=plt.cm.OrRd)
    plt.title('Goal confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

