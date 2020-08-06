"""

Sandbox script

"""

import cv2
import time
import os
import numpy as np
import pickle
from messages import Request, Response
from robots.robot_selector import robot
from BlockBuildingGame import BlockBuildingGame
from BlockBuildingGame2 import BlockBuildingGame2
from BlockObserver import BlockObserver
from CognitiveArchitecture import CognitiveArchitecture
from Skeleton import Skeleton
from ExtraFeatures import ExtraFeatures
import statistics as stat
import tokenize
from L2Node import L2Node
from Buffer import Buffer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# -------------------------------------------------------------------------------------------------------------------- #

# SIMULATION MODE

"""
from simulation.CognitiveArchitecture import CognitiveArchitecture as SimulatedCognitiveArchitecture

datapath = "/home/samuele/Research/datasets/block-building-game/"

cog = SimulatedCognitiveArchitecture()
cog.set_datapath(datapath)
cog.process(reload=False)
"""

"""
buff = Buffer(None)
tokens = [0, 0, 6, 5, 6, 0]
for token in tokens:
    buff.insert(token)
"""


def inspection():
    def calculate_centroid(data):
        arr = np.asarray(data)
        length = arr.shape[0]
        sum_x = np.sum(arr[:, 0])
        sum_y = np.sum(arr[:, 1])
        centroid = sum_x / length, sum_y / length
        return centroid

    cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
    cog.train(reload=True)
    # Cluster correction
    cog.lowlevel.train.clusters[5].centroid = [1.4346541941018343, -0.1571525604543636]
    cog.lowlevel.train.clusters[6].centroid = [1.0692279685864146, -0.02116144880533045]
    # Retrieves the subcluster skeletons
    c5 = cog.lowlevel.train.skeletons_by_cluster(5)
    c6 = cog.lowlevel.train.skeletons_by_cluster(6)
    '''
    print("CLUSTER 5")
    for skeleton in c5:
        print(skeleton.id)
        skeleton.display_fast()
    print("CLUSTER 6")
    '''
    for skeleton in c6:
        print(skeleton.id)
        skeleton.display_fast()

    # Get the skeletons which have to be moved
    skeletons_5_to_6 = [cog.lowlevel.train.get_skeleton_by_id(62)]
    skeletons_6_to_5 = [cog.lowlevel.train.get_skeleton_by_id(90), cog.lowlevel.train.get_skeleton_by_id(133)]
    # Copy the skeletons to the respective new groups
    c5.extend(skeletons_6_to_5)
    c6.extend(skeletons_5_to_6)
    # Delete the skeletons from the old group
    for skeleton in c5:
        if skeleton.id == 62:
            c5.remove(skeleton)
    for skeleton in c6:
        if skeleton.id == 90 or skeleton.id == 133:
            c6.remove(skeleton)
    # Compute the data sets
    data5 = [skeleton.as_feature(only_extra=True) for skeleton in c5]
    data6 = [skeleton.as_feature(only_extra=True) for skeleton in c6]
    # Calculate PCA to obtain the 2d data though L2Nodes
    node_5 = L2Node(5, data5)
    data5_2d = node_5.dataset2d
    node_6 = L2Node(5, data6)
    data6_2d = node_6.dataset2d
    # Calculate the new centroids
    centroid5 = calculate_centroid(data5_2d)
    centroid6 = calculate_centroid(data6_2d)
    # Output
    print("CENTROID 5: " + str(centroid5))
    print("CENTROID 6: " + str(centroid6))
    pass


#single_block_confusion_matrix()
#multi_block_confusion_matrix()

#img = cv2.imread("/home/samuele/Research/datasets/block-building-game/test/castle-small/frame0001.jpg")
#img = cv2.imread("/home/samuele/Research/models/fullbody.jpg")
#skeleton = Skeleton(img)
#skeleton.plot(save=False)
#skeleton.display(background=True, save=False, savename="fullbody", color=(0,255,255))
#skeleton.cippitelli_norm()
#skeleton.plot(dimensions=2)
#print("Orientation: " + skeleton.orientation_reach())

#print(icub.wait_and_listen_remote())

#bbn = BeliefNetwork("test", "belief/datasets/examples/tricker.csv")
#print(bbn.decision_making('A'))
#bbn_e = BeliefNetwork.create_episodic([bbn], 6)
#print(bbn_e.test_query(prettyTable=True))
#print(bbn_e.decision_making('A'))
#prediction = bbn.belief_estimation('A')
#rel = bbn.get_reliability()
#print("REL: " + str(rel))

"""
# Complete game
time.sleep(2)
bb = BlockBuildingGame(debug=True)
reload = True
if not reload:
    bb.training_phase()
else:
    bb.reload_training()
# Cluster editing for the video
#clusters = bb.cognition.lowlevel.train.clusters
#bb.cognition.lowlevel.train.clusters[1].id = 2
#bb.cognition.lowlevel.train.clusters[2].id = 1
#bb.cognition.lowlevel.train.show_clustering()
bb.playing_phase(point=True)
bb.end()
"""



#robot.action_home()
#robot.action_give()
#robot.say("Take this!")
#robot.action_home()
#robot.action_close()

#bb = BlockBuildingGame(debug=True)
#bb.reload_training()
#bb.playing_phase(point=True)
#bb.end()

#time.sleep(3)
#image = robot.get_camera_frame()
#skeleton = robot.look_for_skeleton(None, 0)
#skeleton.display_fast()

'''
obs = BlockObserver()
img = cv2.imread("./img/blocks2/valid.jpeg")    # valid, invalid, incomplete, empty
sequence = obs.detect_sequence(img)
print(sequence)
print("Valid? " + str(obs.validate_sequence()))
print("Number of blocks: " + str(len(sequence)))
obs.display()
'''

'''
time.sleep(2)
robot.say("Go!")
for i in range(10):
    image = robot.get_camera_frame()
    cv2.imwrite("img/frames/" + str(i) + ".jpg", image)
    time.sleep(1)
robot.say("End")
'''


'''
# Training only
bb = BlockBuildingGame2(debug=True, save=True)
bb.training_phase()
bb.cognition.lowlevel.train.summarize_training()
'''

'''
# Saves skeleton images to disk
cog = CognitiveArchitecture()
cog.train(reload=True)
for skeleton in cog.lowlevel.train.skeletons:
    cv2.imwrite("img/experiment2/trainingset/" + str(skeleton.id) + ".jpg", skeleton.origin)
'''

'''
cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
cog.train(reload=False)
cog.lowlevel.train.summarize_training()
print("Done")
'''

'''
cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
cog.train(reload=True)
cog.lowlevel.train.summarize_training()
l2nodes = []
clusters = cog.lowlevel.train.clusters
skeleton_ids_1 = clusters[1]
data1 = [skeleton.as_feature(only_extra=True) for skeleton in cog.lowlevel.train.skeletons if skeleton.id in skeleton_ids_1.skeleton_ids]
l2n1 = L2Node(1, data1)
l2nodes.append(l2n1)
skeleton_ids_2 = clusters[2]
data2 = [skeleton.as_feature(only_extra=True) for skeleton in cog.lowlevel.train.skeletons if skeleton.id in skeleton_ids_2.skeleton_ids]
l2n2 = L2Node(2, data2)
l2nodes.append(l2n2)
pickle.dump(l2nodes, open("objects/" + "l2nodes.p", "wb"))
print("Done!")
#cog.read_intention(simulation=True)
#print("Done")
'''

'''
basepath = "/home/samuele/Research/PyCharm Projects/DeCIFER/img/experiment2/test-frames/{}.jpg"
names = ["000", "B", "G", "O", "R"]
for name in names:
    print("---{}---".format(name))
    path = basepath.format(name)
    img = cv2.imread(path)
    s = Skeleton(img, 0)
    f = s.as_feature()
    print(f)
'''

'''
cog = CognitiveArchitecture(debug=True, offline=True, persist=False)
cog.train(reload=True)

# Cluster correction
cog.lowlevel.train.clusters[5].centroid = [1.4346541941018343, -0.1571525604543636]
cog.lowlevel.train.clusters[6].centroid = [1.0692279685864146, -0.02116144880533045]

cog.lowlevel.train.summarize_training()
cog.read_intention(simulation=False)
'''

#robot.action_display("eyes")

'''
game = BlockBuildingGame2(debug=True, save=False, fixed_goal=True, naive_trust=True)
game.reload_training()
# Cluster correction
game.cognition.lowlevel.train.clusters[5].centroid = [1.4346541941018343, -0.1571525604543636]
game.cognition.lowlevel.train.clusters[6].centroid = [1.0692279685864146, -0.02116144880533045]
game.play_demo_notrust()
game.end()
'''

'''
for color in ["blue", "orange", "red", "green"]:
    robot.action_home()
    robot.action_midpose()
    robot.action_midpose_high()
    robot.action_take(color)
    robot.action_midpose_high()
    robot.action_give()
    robot.action_home()
'''

#robot.action_home()

"""
from colorfilters import HSVFilter

obs = BlockObserver()
#img_path = "img/experiment2/trainingset/RBGO/130.jpg"
#img_path = "img/experiment2/trainingset/OGBR/67.jpg"
#img_path = "img/experiment2/trainingset/ORBG/47.jpg"
img_path = "/home/samuele/Research/PyCharm Projects/DeCIFER/img/experiment2/trainingset/BGOR/38.jpg"
#img_path = "img/bgor.png"

img = cv2.imread(img_path)

#window = HSVFilter(img)
#window.show()
#print("Image filtered in HSV between {" + str(window.lowerb) + "} and {" + str(window.upperb) + "}.")

print(obs.process(img))
#obs.display()
"""

game = BlockBuildingGame2(debug=True, save=False, simulation=True, trust=True)
game.reload_training()
game.playing_phase_trust()
game.end()

