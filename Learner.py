"""

This class is in charge of learning intentions from training examples, managing everything that goes from skeleton
extraction to cluster generation.

"""

from Skeleton import Skeleton, NoHumansFoundException
from Cluster import Cluster
from Intention import Intention
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.patheffects as path_effects
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
import pyclustering.cluster.xmeans as pyc
from pyclustering.utils import draw_clusters
from pyclustering.cluster.silhouette import silhouette
import os
import pickle
import csv
import math
from robots.robot_selector import robot
import cv2
import subprocess
import time
from L2Node import L2Node
import copy


class Learner:
    DIMENSIONS = 2              # Currently working with 2D skeletons
    PCA_DIMENSIONS = 2           # Dimensionality reduction target dimensions
    latest_index = None         # This is used in L2 cluster identification assignment

    def __init__(self, debug=False, persist=False):
        self.skeletons = []     # Observed skeletons
        self.offsets = []       # Splits the dataset in sequences
        self.dataset = []       # 20-D dataset
        self.dataset2d = []     # 2-D dataset
        self.clusters = []      # Clusters
        self.l2nodes = []       # Progressive Clustering L2 nodes
        self.intentions = []    # Intentions
        self.goal_labels = []   # Goal labels
        self.pca = None         # Trained parameters of a PCA model
        self.ax = None          # Plotting purpose
        self.debug = debug      # If true, vocal commands will be substituted by keyboard input
        self.persist = persist  # If True, saves data structures to disk

    # --- INITIALIZATION METHODS --- #

    # Acquires and processes the training data
    # Can be a new, fresh learning or the update of an existing knowledge base
    def learn(self, new=True, savedir="objects/"):
        if new:
            self.generate_skeletons()
        else:
            self.update_knowledge()
        self.generate_dataset()
        self.do_pca()
        sep, dim = self.generate_clusters()
        self.generate_intentions()
        self.postprocess_intenentions(sep, dim-1)
        if self.persist:
            self.save(savedir)

    # Reload already computed Controller data
    def reload_data(self, path="objects/"):
        try:
            print("[DEBUG] Reloading from " + str(path))
            self.load(path)
        except Exception:
            print("Error: failed to load Controller data.")
            quit(-1)

    # --- SAVE AND LOAD METHODS --- #

    # Saves the objects in binary format
    def save(self, savedir="objects/"):
        if savedir[-1] != '/':
            savedir += '/'
        # Verify the existence of the saving directory
        dir = os.path.dirname(savedir)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Saves each attribute of the class
        pickle.dump(self.skeletons, open(savedir + "skeletons.p", "wb"))
        pickle.dump(self.dataset, open(savedir + "dataset.p", "wb"))
        pickle.dump(self.dataset2d, open(savedir + "dataset2d.p", "wb"))
        pickle.dump(self.clusters, open(savedir + "clusters.p", "wb"))
        pickle.dump(self.offsets, open(savedir + "offsets.p", "wb"))
        pickle.dump(self.intentions, open(savedir + "intentions.p", "wb"))
        pickle.dump(self.goal_labels, open(savedir + "goal_labels.p", "wb"))
        pickle.dump(self.pca, open(savedir + "pca.p", "wb"))
        pickle.dump(self.ax, open(savedir + "ax.p", "wb"))
        print("[DEBUG] model saved in " + str(savedir))

    # Loads the objects from binary format
    def load(self, path="objects/"):
        if path[-1] != '/':
            path += '/'
        # Loads each attribute of the class
        self.skeletons = pickle.load(open(path + "skeletons.p", "rb"))
        self.dataset = pickle.load(open(path + "dataset.p", "rb"))
        self.dataset = pickle.load(open(path + "dataset.p", "rb"))
        self.dataset2d = pickle.load(open(path + "dataset2d.p", "rb"))
        self.clusters = pickle.load(open(path + "clusters.p", "rb"))
        self.offsets = pickle.load(open(path + "offsets.p", "rb"))
        self.intentions = pickle.load(open(path + "intentions.p", "rb"))
        self.goal_labels = pickle.load(open(path + "goal_labels.p", "rb"))
        self.pca = pickle.load(open(path + "pca.p", "rb"))
        self.ax = pickle.load(open(path + "ax.p", "rb"))

    # --- METHODS --- #

    # Performs the learning by demonstration task
    def generate_skeletons(self):
        # Loops for all the goals to be learned
        finished = False
        i = 0
        while not finished:
            robot.say("Please, start demonstrating.")
            if robot.__class__.__name__ == "Sawyer":
                #robot.say("[watching]")
                robot.action_display("eyes")
            # Learns a single goal
            skeletons, goal_name = robot.record_goal(i, fps=2, debug=self.debug)
            self.skeletons.extend(skeletons)    # extend instead of append to avoid nesting lists
            self.goal_labels.append(goal_name)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
            i += len(skeletons)
            robot.say("Do you want to show me another goal?")
            while True:
                if not self.debug:
                    response = robot.wait_and_listen()
                else:
                    response = robot.wait_and_listen_dummy()
                if robot.recognize_commands(response, listenFor="NO"):
                    robot.say("All right, thanks for showing me.")
                    finished = True
                    break
                elif robot.recognize_commands(response, listenFor="YES"):
                    robot.say("Ok then, let's continue.")
                    break
                else:
                    robot.say("Sorry, I didn't understand. Can you repeat?")

    # Debug mode for skeleton acquisition: input from file rather than from robot eyes
    # Assumes each goal is contained in a separated subdir named as the goal itself
    def offline_learning(self, path="img/experiment2/trainingset/", volume=1, savedir="objects/offline/"):
        tic = time.time()
        id = 0
        for folder in os.listdir(path):
            print("---Processing folder: " + folder)
            basename = os.path.join(path, folder) + "/"
            # Load images from desired folder
            # This line of code creates an ordered list of filenames, even if the latter are not alphabetically
            # orderable (because front padding is missing, i.e. RGB_1, RGB_2 instead of RGB_1, RGB_10...)
            sorted_files = [str(basename + f.decode("utf-8")) for f in
                            subprocess.check_output("ls " + basename + "/ | sort -V", shell=True).splitlines()]
            # Only consideres 1/volume (default 1/5) of the data
            sorted_files = sorted_files[::volume]
            images = np.empty(len(sorted_files), dtype=object)
            for n in range(0, len(sorted_files)):
                #print("Veryfing: " + str(sorted_files[n]))
                images[n] = cv2.imread(sorted_files[n])
            # Create skeletons for all of them
            skeletons = []
            for image in images:
                try:
                    skeleton = Skeleton(image, id)
                    skeletons.append(skeleton)
                    id += 1
                except NoHumansFoundException:
                    continue
            self.skeletons.extend(skeletons)
            self.goal_labels.append(folder)
            self.offsets.append(len(skeletons) + (0 if len(self.offsets) == 0 else max(self.offsets)))
        dt = time.time() - tic
        print("\n\nProcessing completed. Elapsed time: " + str(dt / 60) + " minutes.")
        # Continue with normal learning phases
        self.generate_dataset()
        self.do_pca()
        self.generate_clusters()
        self.generate_intentions()
        if self.persist:
            self.save(savedir)

    # Builds the dataset feature matrix of dimension (n x 20)
    def generate_dataset(self, feature_scaling=False):
        dim = len(self.skeletons[0].as_feature(add_extra=False))   # Number of columns needed
        # Creates the dataset array
        dataset = np.zeros(shape=(1, dim))
        for skeleton in self.skeletons:
            # skeleton.display()
            dataset = np.vstack((dataset, skeleton.as_feature(add_extra=False)))
        # Removes the first, empty row
        self.dataset = dataset[1:]
        # Optional feature scaling (standardization)
        if feature_scaling:
            self.dataset = StandardScaler().fit_transform(self.dataset)

    # Performs dimensionality reduction from n-D to 2-D through PCA
    def do_pca(self):
        # PCA to reduce dimensionality to 2D
        self.pca = PCA(n_components=Learner.PCA_DIMENSIONS).fit(self.dataset)
        self.dataset2d = self.pca.transform(self.dataset).tolist()

    # Dynamical hyperparameter search for XMeans. Finds the best value of tolerance based on silhouette score
    def dynamical_hyperparameter_search(self, data, initial_centers, debug=False):
        best_tolerance = 0
        best_score = 0
        best_n_clusters = 0
        print("Dynamical hyperparameter search in progress...")
        for tolerance in np.arange(0.001, 0.3, 0.001):
            xmeans_instance = pyc.xmeans(data, initial_centers, ccore=True, kmax=20, tolerance=tolerance,
                                         criterion=pyc.splitting_type.BAYESIAN_INFORMATION_CRITERION)
            xmeans_instance.process()
            cluster_lists = xmeans_instance.get_clusters()
            score = np.average(silhouette(list(self.dataset2d), cluster_lists).process().get_score())
            if debug:
                print("Evaluating tolerance " + str(tolerance) + "... score " + str(score) + "(" + str(len(cluster_lists)) + " clusters)")
            if score > best_score:
                best_tolerance = tolerance
                best_score = score
                best_n_clusters = len(cluster_lists)
        print("BEST: " + str(best_tolerance) + " with score " + str(best_score) + "( " + str(best_n_clusters) + " clusters)")
        return best_tolerance

    # XMeans clustering
    # Parameter base_id is used in second-level clustering to determine the proper nomenclature of the clusters
    def xmeans_clustering(self, data, use_BIC=True, base_id=None, removal_threshold=3, reference=None):
        # initial centers with K-Means++ method
        initial_centers = kmeans_plusplus_initializer(list(self.dataset2d), Learner.PCA_DIMENSIONS).initialize()
        # Dynamical search of the best hyperparameter
        tolerance = self.dynamical_hyperparameter_search(self.dataset2d, initial_centers)
        if use_BIC:
            criterion = pyc.splitting_type.BAYESIAN_INFORMATION_CRITERION
        else:
            criterion = pyc.splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
        # create object of X-Means algorithm that uses CCORE for processing (default tolerance: 0.025)
        xmeans_instance = pyc.xmeans(data, initial_centers, ccore=True, kmax=20, tolerance=tolerance, criterion=criterion)
        # run cluster analysis
        xmeans_instance.process()
        # obtain results of clustering
        centers = xmeans_instance.get_centers()
        cluster_lists = xmeans_instance.get_clusters()
        # Calculate Silhouette score for each point and averages it
        score = np.average(silhouette(list(self.dataset2d), cluster_lists).process().get_score())
        clusters = []
        colors = Cluster.get_colors(len(centers))
        # Outlier removal: simply discard clusters which are too small
        for i, cluster_list in enumerate(cluster_lists):
            if len(cluster_list) < removal_threshold:
                cluster_lists.pop(i)
                centers.pop(i)
                print("[DEBUG] I have removed cluster " + str(cluster_list) + " from the final cluster list.")
        for i in range(len(centers)):
            # Determines the cluster nomenclature
            if base_id is not None:
                #id = base_id + (i+1) / 10.0        # e.g. cluster id 2.1, 2.2 ...
                id = Learner.latest_index + 1
                Learner.latest_index += 1
                level = 2
            else:
                id = i
                level = 1
            cl = copy.deepcopy(cluster_lists)   # I need to make a copy because of the id fixation that will come after
            c = Cluster(centers[i], cl[i], id, colors[i], level)
            clusters.append(c)
        # If this is a L2 clustering process, skeleton ids must be fixed
        # xmeans.get_clusters returns id wrt the passed dataset. We need ids wrt the global dataset
        if base_id is not None and reference is not None:
            for cluster in clusters:
                for i in range(len(cluster.skeleton_ids)):
                    value = cluster.skeleton_ids[i]
                    cluster.skeleton_ids[i] = reference[value]
        self.ax = draw_clusters(data, cluster_lists, display_result=True)
        return clusters, score, [centers, cluster_lists]

    # Performs X-Means clustering on the provided dataset
    def generate_clusters(self):
        final_clusters = []
        # Perform xmeans clustering on the sole skeletal data
        print("L1 clustering")
        clusters, _, _ = self.xmeans_clustering(self.dataset2d, use_BIC=True)       # L1 clustering
        # todo debug... remove
        if len(clusters) != 3:
            print("[DEBUG] Re-clustering...")
            self.generate_clusters()
            return
        final_clusters.extend(clusters)
        '''
        # Display
        self.clusters = final_clusters
        # Create image plot
        for skeleton in self.skeletons:
            im = OffsetImage(skeleton.img, zoom=0.38)
            coordinates = self.dataset2d[skeleton.id]
            cluster_id = self.find_cluster_id(skeleton.id, limit=True)
            # Sanity check
            if cluster_id is None:
                print("Error! Couldn't find cluster in which a skeleton belongs")
                raise RuntimeError
            else:
                color = self.clusters[cluster_id].color
                ab = AnnotationBbox(im, coordinates, bboxprops=dict(edgecolor=color))
                ab.set_zorder(2)
                self.ax.add_artist(ab)
        # Plots the clusters centroids
        for cluster in self.clusters:
            x = cluster.centroid[0]
            y = cluster.centroid[1]
            text = self.ax.text(x, y + 0.015, cluster.id, fontsize=40, color='white')
            # Adds the black border around the text
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
            plt.scatter(x, y, zorder=3, marker='o', s=10, c=cluster.color, edgecolors='black', linewidths='2')
        plt.show()
        '''
        # Define the cluster which contains the neutral posture (the largest one)
        other_clusters = clusters.copy()
        neutral_cluster = max(other_clusters, key=lambda x: len(x.skeleton_ids))
        other_clusters.remove(neutral_cluster)
        Learner.latest_index = max([cluster.id for cluster in clusters])
        # For each one of the secondary clusters, re-cluster using only the extra features
        for parent_cluster in other_clusters:
            # Collect the data
            data = [skeleton.as_feature(only_extra=True) for skeleton in self.skeletons if skeleton.id in parent_cluster.skeleton_ids]
            # Store this subdataset as an L2Node to be able to perform L2 inference during the intention reading phase
            new_l2_node = L2Node(parent_cluster.id, data)
            self.l2nodes.append(new_l2_node)
            data2d = new_l2_node.dataset2d
            # Perform clustering
            print("L2 clustering")
            secondary_clusters, _, pyc_data = self.xmeans_clustering(data2d, use_BIC=False, base_id=parent_cluster.id, reference=parent_cluster.skeleton_ids)
            final_clusters.extend(secondary_clusters)
            parent_cluster.descendants = secondary_clusters
            self.ax = draw_clusters(data2d, pyc_data[1], display_result=False)
            # todo temporary: plot sub-cluster pictures
            for skeleton in [skeleton for skeleton in self.skeletons if skeleton.id in parent_cluster.skeleton_ids]:
                im = OffsetImage(skeleton.img, zoom=0.38)
                coordinate_index = parent_cluster.skeleton_ids.index(skeleton.id)  # Data and skeletons are not aligned, I must fetch the id specifically
                coordinates = data2d[coordinate_index]
                cluster_id = self.find_cluster_id(skeleton.id)
                #if cluster_id is not None:
                color = 'red' if skeleton.id in secondary_clusters[0].skeleton_ids else 'blue'
                ab = AnnotationBbox(im, coordinates, bboxprops=dict(edgecolor=color))
                ab.set_zorder(2)
                self.ax.add_artist(ab)
            # Plots the clusters centroids
            for sub_cluster in secondary_clusters:
                x = sub_cluster.centroid[0]
                y = sub_cluster.centroid[1]
                text = self.ax.text(x, y + 0.015, sub_cluster.id, fontsize=40, color='white')
                # Adds the black border around the text
                text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
                plt.scatter(x, y, zorder=3, marker='o', s=10, c=sub_cluster.color, edgecolors='black', linewidths='2')
            plt.show()
            #self.show_clustering()
        # Recolor all the clusters
        colors = Cluster.get_colors(len(final_clusters))
        for i in range(0, len(final_clusters)):
            final_clusters[i].color = colors[i]
        self.clusters = final_clusters
        # Returns the neutral cluster id and the number of non-parent clusters, to be used for post-processing
        return neutral_cluster.id, len([cluster.id for cluster in final_clusters if cluster.descendants is None])

    # Fetches a specified cluster by id
    def find_cluster_by_id(self, id):
        for cluster in self.clusters:
            if cluster.id == id:
                return cluster
        return None

    # Gets a cluster set or subset.
    def get_cluster_set(self, level, parent_id=None):
        assert level > 0, "Invalid value for level parameter"
        if level == 1:
            return [cluster for cluster in self.clusters if cluster.level == 1]     # All L1 clusters
        else:
            assert parent_id is not None, "A parent id must be specified for levels 2+"
            parent_cluster = self.find_cluster_by_id(parent_id)
            if parent_cluster is None:
                return []
            else:
                return [cluster for cluster in parent_cluster.descendants]  # Specific L2 subset

    # Find the cluster id containing the skeleton
    def find_cluster_id(self, skeleton_id):
        # Iterates over the L1 clusters
        for l1_cluster in self.get_cluster_set(1):
            if skeleton_id in l1_cluster.skeleton_ids:
                # Once it finds the skeleton, it checks wheter the cluster is a parent (in that case, search must go on)
                if not l1_cluster.is_parent():
                    return l1_cluster.id
                else:
                    # Search in the specific subset of L2 clusters
                    for l2_cluster in self.get_cluster_set(2, l1_cluster.id):
                        if skeleton_id in l2_cluster.skeleton_ids:
                            return l2_cluster.id
        return None

    # Finds the closest centroid to a new skeletal sample (already in 2D coordinates).
    # A list of clusters in which to search for must be provided.
    def find_closest_centroid(self, sample2d, cluster_set):
        min_distance = float("inf")
        closest_cluster = None
        for cluster in cluster_set:
            dist = math.hypot(sample2d[0] - cluster.centroid[0], sample2d[1] - cluster.centroid[1])
            if dist < min_distance:
                min_distance = dist
                closest_cluster = cluster
        return closest_cluster

    # Retrieves an L2Node with specified id, or None if not found
    def get_l2node(self, id):
        for l2node in self.l2nodes:
            if l2node.id == id:
                return l2node
        return None

    # Saves a dataset as a csv
    def save_csv(self):
        file = open('csv/dataset.csv', 'w')
        with file:
            writer = csv.writer(file)
            writer.writerows(self.dataset)

    # Computes intentions for each training sequence
    def generate_intentions(self):
        # Checks that lengths are equals
        if len(self.goal_labels) != len(self.offsets):
            print("[ERROR] Dimension mismatch between goal label list and offset list.")
            quit(-1)
        # Consider every sequence
        previous = 0
        for offset_index in range(len(self.offsets)):
            intention = Intention()
            for skeleton_index in range(previous, self.offsets[offset_index]):
                # Retrieve the cluster id
                cluster_id = self.find_cluster_id(self.skeletons[skeleton_index].id)
                if cluster_id is not None and (len(intention.actions) == 0 or intention.actions[-1] != cluster_id):
                    intention.actions.append(cluster_id)
            # Create the goal label from pathname
            intention.goal = self.goal_labels[offset_index]
            # Save the computed intention
            self.intentions.append(intention)
            previous = self.offsets[offset_index]

    # Corrects the intentions to account for more structure and noise in the training data
    def postprocess_intenentions(self, separator, dim):
        intentions = self.intentions
        matrix = []
        label_order = None
        for intention in intentions:
            actions = intention.actions
            size = len(actions)
            idx_list = [idx + 1 for idx, val in enumerate(actions) if
                        val == separator]  # Calculates the indexes offsets
            res = [actions[i: j] for i, j in
                   zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))]  # Groups the elements
            res = [[subelt for subelt in elt if subelt != separator] for elt in res]  # Removes the separator value
            res = list(filter(None, res))  # Removes any empty lists
            # Eliminates any irregular data (both value and index will be missing, making it useless
            if len(res) != dim:
                continue
            else:
                label = list(intention.goal)  # Tokenizes the label
                label_order = np.sort(label)  # Redundant, but does its job
                indexes = np.argsort(label)  # Outputs the indexes of the sorted array
                ordered_values = []
                for i in indexes:
                    ordered_values.append(res[i])  # Rebuilds the array based on the ordered indexes
                matrix.append(ordered_values)
        # Count the values
        counters = {'B': 0, 'G': 0, 'O': 0, 'R': 0}
        transposed = np.transpose(matrix)  # Transposes the matrix, so that each row will repesent one letter
        for i in range(len(transposed)):
            row = transposed[i]
            row = [item for sublist in row for item in sublist]  # Flattens the list
            expected_value = stat.mode(row)  # Selects the most common value in each row
            letter = label_order[i]  # Retrieves the "letter" under review
            counters[letter] = expected_value
        # Compute the new, processed intentions as: [separator, value, separator, ... , value, separator]
        for intention in intentions:
            label = list(intention.goal)
            new_action = [separator]
            for letter in label:
                new_action.append(counters[letter])
                new_action.append(separator)
            intention.actions = new_action
        self.intentions = intentions

    # Generates a list containing the intentions in dictionary form (training dataset)
    def make_training_dataset(self):
        dict_list = []
        for intention in self.intentions:
            dict_list.append(intention.as_dict())
        return dict_list

    # Summarizes the results of the training
    def training_result(self):
        print("SUMMARY OF TRAINING")
        print("Clusters: " + str(len(self.clusters)))
        print("Intentions:")
        for intention in self.intentions:
            print(str(intention.goal) + " : " + str(intention.actions))

    # Determines the general reach orientation of each cluster
    def cluster_orientation_reach(self):
        output = [''] * len(self.clusters)
        for cluster in self.clusters:
            orientations = {
                "left": 0,
                "right": 0,
                "center": 0
            }
            for skeleton_id in cluster.skeleton_ids:
                orientations[self.skeletons[skeleton_id].orientation_reach()] += 1
            output[cluster.id] = max(orientations, key=orientations.get)    # Gets the key with highest value
        return output

    # Updates the knowledge base with a new goal
    # If the name of the goal is already know, replaces the previous training received
    def update_knowledge(self):
        # Fetch the last id of the dataset
        last_id = max([skeleton.id for skeleton in self.skeletons])
        # Learn one new goal
        print("[DEBUG] Acquiring a new goal...")
        new_skeletons, new_goal = robot.record_goal(last_id+1, fps=2, debug=self.debug)
        # Check to see if the goal is a new one or a replacement
        if new_goal in self.goal_labels:
            # Get the id of the old goal which is being replacing
            discard_id = self.goal_labels.index(new_goal)
            self.dataset_shift(discard_id, new_skeletons)
        else:
            self.skeletons.extend(new_skeletons)  # extend instead of append to avoid nesting lists
            self.goal_labels.append(new_goal)
            self.offsets.append(len(new_skeletons) + max(self.offsets))
        # Re-train the system
        self.learn(new=False, savedir=None)

    # Replaces a skeleton dataset for a specified goal. The existing skeletons associated to the training of that task
    # are eliminated, the new ones are appended to the set and all the ids are shifted left to guarantee continuity
    def dataset_shift(self, discard_id, new_set):
        # Sanity checks
        assert 0 <= discard_id <= len(self.offsets)-1, "[ERROR] discard_id must be a valid index of the offsets list."
        assert all(isinstance(n, Skeleton) for n in new_set), "[ERROR] new_set must contain Skeleton types."
        # Delete the deprecated elements from the dataset
        discard_offset = self.offsets[discard_id]
        if discard_id == 0:
            offset_begin = 0
        else:
            offset_begin = self.offsets[discard_id - 1]
        for x in range(offset_begin, discard_offset):
            self.skeletons[x] = None
        self.skeletons = [x for x in self.skeletons if x is not None]
        # Extend with the new data
        self.skeletons.extend(new_set)
        # Shift back the ids of the elements
        for i in range(len(self.skeletons)):
            if self.skeletons[i].id != i:
                self.skeletons[i].id = i
        # Fix the offsets
        start = 0
        for i in range(len(self.offsets)):      # Offsets to subset lengths
            temp = self.offsets[i]
            self.offsets[i] -= start
            start = temp
        self.offsets.extend([len(new_set)])     # Addition of the new subset length
        self.offsets.pop(discard_id)            # Removal of the unused subset length
        start = 0
        for i in range(len(self.offsets)):      # Subset lengths to offsets
            self.offsets[i] += start
            start = self.offsets[i]

    # --- DISPLAY METHODS --- #

    # Displays a human-friendly result of the clustering operation
    def show_clustering(self, just_dots=False):
        # Sanity check
        if self.ax is None:
            print("Error: must generate clusters before trying to display them.")
            raise RuntimeError
        if not just_dots:
            # Create image plot
            for skeleton in self.skeletons:
                im = OffsetImage(skeleton.img, zoom=0.38)
                coordinates = self.dataset2d[skeleton.id]
                cluster_id = self.find_cluster_id(skeleton.id)
                # Sanity check
                if cluster_id is None:
                    print("Error! Couldn't find cluster in which a skeleton belongs")
                    raise RuntimeError
                else:
                    color = self.clusters[cluster_id].color
                    ab = AnnotationBbox(im, coordinates, bboxprops=dict(edgecolor=color))
                    ab.set_zorder(2)
                    self.ax.add_artist(ab)
        # Plots the clusters centroids
        for cluster in self.clusters:
            x = cluster.centroid[0]
            y = cluster.centroid[1]
            text = self.ax.text(x, y + 0.015, cluster.id, fontsize=40, color='white')
            # Adds the black border around the text
            text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])
            plt.scatter(x, y, zorder=3, marker='o', s=10, c=cluster.color, edgecolors='black', linewidths='2')
        plt.show()

    # Plots the whole action representations for each goal, with the associated skeletons
    def plot_goal(self, verbose=False):
        plt.clf()           # |
        plt.cla()           # | Flush previous plots
        plt.close()         # |
        # Font sizes
        plt.rcParams.update({'axes.titlesize': 24})
        plt.rcParams.update({'axes.labelsize': 14})
        plt.rcParams.update({'xtick.labelsize': 14})
        plt.rcParams.update({'ytick.labelsize': 14})
        last_id = -1
        skeleton_list = []
        intention_id = 0        # Used to retrieve the correct offset value for each intention
        previous_offset = 0
        for goal in self.intentions:
            skeleton_count = 0
            if verbose:
                print("\nGoal: \"" + goal.goal + "\"\nClusters: " + str(goal.actions))
            # Phase 1: obtain the skeletons
            offset = self.offsets[intention_id]
            for cluster_id in goal.actions:
                sk_id = [id for id in self.clusters[cluster_id].skeleton_ids
                         if (id > last_id and previous_offset <= id < offset)][0]
                skeleton_list.append(sk_id)
                last_id = sk_id
                skeleton_count += 1
            if verbose:
                print("Skeleton ids: " + str(skeleton_list) + "\nCount: " + str(skeleton_count))
            # Phase 2: plot
            fig = plt.figure()
            # Start plotting each subfigure
            for i in range(skeleton_count):
                skeleton = self.skeletons[skeleton_list[i]]
                # set up the axes for the i-th plot
                ax = fig.add_subplot(1, skeleton_count, i+1)
                connections = [['Head', 'Neck'],
                               ['Neck', 'Torso'],
                               ['Neck', 'RElbow'],
                               ['Neck', 'LElbow'],
                               ['RElbow', 'RWrist'],
                               ['LElbow', 'LWrist'],
                               ['Torso', 'RKnee'],
                               ['Torso', 'LKnee'],
                               ['RKnee', 'RAnkle'],
                               ['LKnee', 'LAnkle']]
                nonmissing_kp = skeleton.nonmissing_keypoints(with_torso=True)
                array = skeleton.keypoints_to_array(nonmissing_kp)
                x = array[:, 0]
                y = array[:, 1]
                ax.set_xlabel('')
                ax.set_ylabel('')
                # Connect keypoints to form skeleton
                for p1, p2 in connections:
                    if p1 in nonmissing_kp and p2 in nonmissing_kp:
                        start = skeleton.keypoints[p1]
                        end = skeleton.keypoints[p2]
                        ax.plot(np.linspace(start.x, end.x), np.linspace(start.y, end.y), c="blue", marker='|',
                                linestyle='-', linewidth=0.2)
                # Plot the dots
                ax.scatter(x, y, c='b', marker='o', linewidths=5.0)
                plt.title('Skeleton ' + str(skeleton.id) + "\nCluster: " + str(goal.actions[i]))
                plt.grid(True)
                # Puts text
                for label, keypoint in nonmissing_kp.items():
                        ax.text(keypoint.x, keypoint.y, label, None)
            fig.suptitle(str(goal.goal).upper())
            intention_id += 1
            previous_offset = offset
            # Reset
            skeleton_list = []
            last_id = 0
        plt.show()

    # Displays a global summary of the training process
    def summarize_training(self):
        self.training_result()  # Clusters and intentions
        #self.show_clustering()      # Graphical visualization of the clusters
        #self.plot_goal()            # Goals decompositions todo re-enable
