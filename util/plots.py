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
    disp = disp.plot(cmap = plt.cm.OrRd, values_format='.2f')
    #plt.title('Single block confusion matrix')
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
    disp = disp.plot(cmap=plt.cm.OrRd, values_format='.2f')
    #plt.title('Goal confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()


# Success rates bar plot for exp3
def success_rate_bars():
    labels = ['$H_1$', '$H_2$', '$H_3$', '$H_4$']
    men_means = [0, .4, .4, .4]
    women_means = [.66, .82, .82, .5]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='No trust')
    rects2 = ax.bar(x + width/2, women_means, width, label='Trust')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of success')
    ax.set_title('Success rates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim([0, 1])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    plt.savefig('img/success_rates.png', dpi=300)
    #plt.show()


# Reads trust values from csv logs
def read_data_from_csv(id):
    import csv
    values = []
    values.append(0.399999999999999)    # Init trust
    with open('logs/exp03/csv/h' + str(id) + '.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        for row in csv_reader:
            values.append(round(float(row[4]), 3))
    return values


# Trust dynamics
def trust_dynamics():
    fig, axs = plt.subplots(2, 2)
    #fig.suptitle('Trust dynamics')

    size = 13

    fig.text(0.51, 0.04, 'Turn Number', ha='center', fontsize=size)
    fig.text(0.08, 0.5, 'Trust Factor', va='center', rotation='vertical', fontsize=size)

    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    x = range(101)

    for i in range(4):
        y = read_data_from_csv(i)
        axes[i].plot(x, y, colors[i])
        axes[i].set_title('$H_' + str(i+1) +'$')
        #axes[i].set(xlabel='Turn', ylabel='Trust value')
        axes[i].grid()
        axes[i].set_ylim([-0.15, 0.55])
        axes[i].axhline(y=0.0, color='k', linestyle='dashed', lw=0.8)

    plt.savefig('img/trust_dynamics.png', dpi=300)
    #plt.show()


# --- Experiment 3, R1 ---

class simulatedHuman:
    def __init__(self, id, nta, ta, std):
        self.id = '$H_{}$'.format(id)
        self.mean = ta
        self.std = std
        self.nta = nta


# Success rates bar plot for exp3
def success_rate_bars_r1():
    # Data creation
    humans = []
    '''
    # Old data
    humans.append(simulatedHuman(1, 0, .66, 0))
    humans.append(simulatedHuman(2, .4, .82, 0))
    humans.append(simulatedHuman(3, .4, .82, 0))
    humans.append(simulatedHuman(4, .4, .54, .02))
    humans.append(simulatedHuman(5, .64, .8, 0))
    humans.append(simulatedHuman(6, .16, .62, .01))
    '''
    '''
    humans.append(simulatedHuman(1, 0, .97, 0))
    humans.append(simulatedHuman(2, .5, .95, 0))
    humans.append(simulatedHuman(3, .5, .97, 0))
    humans.append(simulatedHuman(4, .5, .60, 0.089))
    humans.append(simulatedHuman(5, .8, .8, 0))
    humans.append(simulatedHuman(6, .2, 0.95, 0.009))
    '''

    humans.append(simulatedHuman(1, 0, .97, 0))
    humans.append(simulatedHuman(2, .5, .5, 0))
    humans.append(simulatedHuman(3, .5, .97, 0))
    #humans.append(simulatedHuman(4, .5, .61, 0.13))
    humans.append(simulatedHuman(4, .5, .66, 0.15)) # 20 trials instead of 10
    humans.append(simulatedHuman(5, .8, .8, 0))
    humans.append(simulatedHuman(6, .2, 0.95, 0.012))
    humans.append(simulatedHuman(7, .8, .81, .01))

    labels = []
    nta_sr = []
    ta_sr = []
    err = []

    for human in humans:
        labels.append(human.id)
        nta_sr.append(human.nta)
        ta_sr.append(human.mean)
        err.append(human.std)

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, nta_sr, width, label='No trust')
    rects2 = ax.bar(x + width/2, ta_sr, width, yerr=err, label='Trust')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage of success')
    #ax.set_title('Success rates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.])
    ax.legend()
    ax.set_ylim([0, 1.1])

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    fig.tight_layout()
    #plt.savefig('img/success_rates_r1.png', dpi=300)
    plt.show()


# Trust dynamics
def trust_dynamics_r1_randomized(id):
    assert 3 <= id <= 6, "Are you sure the ID is a valid one?"

    if id == 6:
        print("WARNING! Have you set initial trust to negative?")

    # Reads trust values from csv logs
    def read_data_from_csv_r1_randomized(id, sequence):
        import csv
        values = []
        values.append(0.399999999999999)  # Init trust
        with open('logs/exp03-R1/H' + str(id) + '/' + str(sequence) + '.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                values.append(round(float(row[4]), 3))
        return values

    fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True)

    #fig.suptitle('Trust dynamics')

    size = 13

    fig.text(0.51, 0.04, 'Turn Number', ha='center', fontsize=size)
    fig.text(0.08, 0.5, 'Opinion', va='center', rotation='vertical', fontsize=size)

    #axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[0, 4],axs[1, 0], axs[1, 1], axs[1, 2]], axs[0, 3], axs[0, 4]]
    axes = [axs[0, 0], axs[0, 1], axs[0, 2], axs[0, 3], axs[0, 4], axs[1, 0], axs[1, 1], axs[1, 2], axs[1, 3], axs[1, 4]]

    colors = {3: 'tab:blue',
              4: 'tab:green',
              5: 'tab:red',
              6: 'tab:orange'}
    color = colors[id]
    #colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    x = range(101)

    for i in range(10):
        y = read_data_from_csv_r1_randomized(id, i)
        axes[i].plot(x, y, color)
        axes[i].set_title(str(i+1))
        #axes[i].set(xlabel='$H_'+str(id)+'$')
        axes[i].grid()
        axes[i].set_ylim([-1.1, 1.1])
        axes[i].axhline(y=0.0, color='k', linestyle='dashed', lw=0.8)

    #plt.savefig('img/trust_dynamics_' + str(id) + '_r1.png', dpi=300)
    plt.show()


# Trust dynamics
def trust_dynamics_r1_deterministic():

    # Reads trust values from csv logs
    def read_data_from_csv_r1_deterministic(id):
        import csv
        values = []
        values.append(0.399999999999999)  # Init trust
        with open('logs/exp03-R1/H' + str(id) +'/0.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            for row in csv_reader:
                values.append(round(float(row[4]), 3))
        return values

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True)

    size = 13

    fig.text(0.51, 0.04, 'Turn Number', ha='center', fontsize=size)
    fig.text(0.08, 0.5, 'Opinion', va='center', rotation='vertical', fontsize=size)

    #axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    x = range(101)

    for i in range(3):
        y = read_data_from_csv_r1_deterministic(i)
        axes[i].plot(x, y, colors[i])
        axes[i].set_title('$H_' + str(i+1) +'$')
        #axes[i].set(xlabel='Turn', ylabel='Trust value')
        axes[i].grid()
        axes[i].set_ylim([-1.1, 1.1])
        #axes[i].set_xlim([0, 101])
        axes[i].axhline(y=0.0, color='k', linestyle='dashed', lw=0.8)

    #plt.savefig('img/trust_dynamics_r1_deterministic.png', dpi=300)
    plt.show()
