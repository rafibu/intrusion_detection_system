import copy
from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor
from os.path import isfile
from random import random

import numpy as np
from scipy.spatial.distance import euclidean
from fcmeans import FCM
from numpy import ndarray
from scipy.optimize import minimize
from scipy.spatial import KDTree
from sklearn.metrics.pairwise import rbf_kernel

from classification.classifier import Classifier
from classification.iftsvm_classifier import split_samples, create_kernel_matrix


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def create_random_population(population_size: int, lower_bound: [], upper_bound: []):
    """
    Initializes a random population of bees as scouts
    :param population_size: number of bees
    :param lower_bound: array of all lower bounds in each dimension
    :param upper_bound: array of all upper bounds in each dimension
    :return: a dictionary which holds the bees
    """
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)
    bees = [(np.inf, lower_bound + [random() for _ in range(len(upper_bound))] * (upper_bound - lower_bound), 0) for _
            in range(int(population_size / 2))]
    return {'employed': bees, 'onlooker': int(population_size / 4), 'scout': int(population_size / 4)}


def create_fcm(clusters: int, bees: ndarray) -> FCM:
    """
    Create a Fuzzy C-means clustering with the given amount of clusters
    :param clusters: number of clusters
    :param bees: observed bees
    """
    fcm = FCM(n_clusters=clusters)
    fcm.fit(bees)
    return fcm


def find_best_fcm(fcms, bees, method, positive_test_samples, negative_test_samples, kernel_matrix):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_fcm_score, bees, fcm, method, kernel_matrix, negative_test_samples, positive_test_samples) for fcm in fcms]
    wait(futures, return_when='FIRST_EXCEPTION')
    scores = [f.result() for f in futures]
    return min(scores, key=lambda s: s[0])[1]


def calculate_fcm_score(bees, fcm, method, kernel_matrix, negative_test_samples, positive_test_samples):
    classification_tree = KDTree(fcm.centers)
    dist = [classification_tree.query(p) for p in bees]
    radii = [0 for _ in range(len(fcm.centers))]
    for d in dist:
        if radii[d[1] - 1] < d[0]:
            radii[d[1] - 1] = d[0]
    temp = ABCClassifier(method=method)
    temp.radii = radii
    temp.classification_tree = classification_tree
    temp.kernel_matrix = kernel_matrix
    score = sum([1 - temp.classify(sample) for sample in positive_test_samples]) + sum(
        [temp.classify(sample) for sample in negative_test_samples])
    return score, fcm


class BeeHive:
    """
    This class describes a single instance of a bee generation
    It is used to facilitate concurrent training
    """

    def __init__(self, k_d_tree, population_size, cycle_numbers, fit_neighbors, sight, chosen_number, fit_func):
        self.chosen_number = chosen_number
        self.sight = sight
        self.k_d_tree = copy.deepcopy(k_d_tree)
        self.population_size = population_size
        self.cycle_numbers = cycle_numbers
        self.fit_neighbors = fit_neighbors
        self.fit_func = fit_func

    def optimize_positions(self, data_set: []):
        """
        Creates new bees and tries to find optimal positions.
        each bee is a tuple of the current fitness score and the position
        :param data_set: all known positive labels
        :return: positions of the best bees
        """
        lower_bound = np.amin(data_set, axis=0)
        upper_bound = np.amax(data_set, axis=0)
        min_distance = euclidean(np.zeros(len(upper_bound)), (upper_bound - lower_bound) / 1000)
        bees = create_random_population(self.population_size, lower_bound, upper_bound)
        for _ in range(self.cycle_numbers):
            newly_freed = self.employed_bees_phase(bees, min_distance)
            self.unemployed_bees_phase(bees, newly_freed, lower_bound, upper_bound)
        return bees.get('employed')[:self.chosen_number]

    def fit_function(self, position: []):
        """
        Find the number of neighbors within the sight of a bee
        :param position: current position of bee
        :return: negative of the neighbors to facilitate minimize
        """
        if self.fit_func == 'count':
            return sum(self.k_d_tree.query(position, self.fit_neighbors)[0])
        return -self.k_d_tree.query_ball_point(x=position, r=self.sight, return_length=True)

    def employed_bees_phase(self, bees, min_distance):
        """
        Employed bees look around a food source to find the most optimal position
        :param min_distance: minimal distance between two bees
        :param bees: a dict of tuples, where the first element is the fitness score and the second is the current position
        """
        new_bees = []
        freed = 0
        for bee in bees.get('employed'):
            # If two bees are nearly at the same place, only the first one stays, the second become scouts
            skip = False
            if len(new_bees) > 0:
                closest = new_bees[closest_node(bee[1], [b[1] for b in new_bees])]
                if euclidean(closest[1], bee[1]) < min_distance:
                    freed += 1
                    skip = True
            if not skip:
                # Only update the position 4 times to reduce complexity
                if bee[2] < 4:
                    pos = minimize(self.fit_function, bee[1], bounds=[(f - self.sight, f + self.sight) for f in bee[1]])
                    new_bees.append((pos.get('fun'), np.array(pos.get('x')), bee[2] + 1))
                else:
                    new_bees.append(bee)
        bees['employed'] = new_bees
        return freed

    def unemployed_bees_phase(self, bees, newly_freed, lower_bound, upper_bound):
        """
        The onlooker bees choose two known food source and look at a point between them to see, if it has a better food
        source
        :param newly_freed: Number of employed bees which were freed in the last round
        :param bees: a dict of tuples, which represent the bees
        :param lower_bound: array of all lower bounds in each dimension
        :param upper_bound: array of all upper bounds in each dimension
        """
        employed = bees.get('employed')
        norm_fact = sum([b[0] for b in employed])
        if norm_fact == 0:
            probabilities = [1 / len(employed) for _ in employed]
        else:
            probabilities = [b[0] / norm_fact for b in employed]
        to_add = []
        for _ in range(bees.get('onlooker')):
            points = np.random.choice(len(employed), size=2, p=probabilities)
            x_1 = employed[points[0]][1]
            x_2 = employed[points[1]][1]
            position = x_1 + random() * (x_1 - x_2)
            fit = self.fit_function(position)
            to_add.append((fit, position, 0))

        lower_bound = np.array(lower_bound)
        upper_bound = np.array(upper_bound)
        for _ in range(bees.get('scout') + newly_freed):
            position = lower_bound + [random() for _ in range(len(upper_bound))] * (upper_bound - lower_bound)
            fit = self.fit_function(position)
            to_add.append((fit, position, 0))

        final_length = len(employed) + newly_freed
        employed.extend(to_add)
        employed.sort(key=lambda b: b[0])
        bees['employed'] = employed[:final_length]


class ABCClassifier(Classifier):
    """
    This class represents the classifier which uses the ABC Algorithm with FCM to create rules which indicate how a
    new sample should be classified.
    First an Artificial Bee Colony is created which flies according to the rules of the Artificial Fish Swarm
    They look for concentrations of samples which lie in the class that is trained for and stay there.
    After a certain time, the location of the bees including their happiness (How close they are to such samples)
    is saved and a new colony is created.
    After some generations, the locations are clustered with the Fuzzy C-means algorithm, into clusters. The
    cluster-center and radius are calculated for each cluster and then used to classify new samples.
    """
    classification_tree: KDTree

    dual_mode = True

    def __init__(self, method='linear', population_size=200, generations=100, cycle_numbers=1000, fit_neighbors=5,
                 sight=5, chosen_number=20, kernel_size=None, fit_function='sight'):
        """
        :param population_size: Size of each bee population, half will be employed bees, the other half will be split
               into onlooker and scout bees
        :param generations: number of how many bee populations should be generated
        :param cycle_numbers: number of steps each population take before stopping
        :param fit_neighbors: number of nearest neighbors to take into account when calculating the fitness function
        :param sight: how far an onlooker bee can see
        :param chosen_number: number of bees which are chosen for each generation
        """
        self.radii = None
        self.population_size = population_size
        self.generations = generations
        self.cycle_numbers = cycle_numbers
        self.fit_neighbors = fit_neighbors
        self.sight = sight
        self.chosen_number = chosen_number
        if method not in ['linear', 'non-linear']:
            raise Exception('method should be either linear of non-linear')
        self.method = method
        self.kernel_matrix = None
        self.kernel_size = kernel_size
        self.fit_function = fit_function

    def train(self, data_set: [], positive_labels: [int]):
        """
        Trains the model according to the ABC-AFS-FCM method through the following steps:
            1. Initialize random Bee population
            2. Employed Bee phase: Bees with the highest concentration of food try to optimize position within
               their neighborhood
            3. Onlooker Bee phase: Bees with the worst food source abandon it and try to find better ones depending
               on the already known ones
            4. Scout Bee phase: scouts fly randomly and try to find better food sources
            5. Save best positions and repeat
            6. Use Fuzzy C-means clustering to find rules for classifying samples
        :param positive_labels: the labels which represent the searched label
        :param data_set: set of normalized data with only the important features present
        """
        positive_samples, negative_samples = split_samples(data_set, positive_labels)
        postive_test = np.array_split(positive_samples, 10)[0]
        negative_test = np.array_split(negative_samples, 10)[0]
        if self.method == 'non-linear':
            self.kernel_matrix = create_kernel_matrix(positive_samples, negative_samples, self.kernel_size)
            positive_samples = rbf_kernel(positive_samples, self.kernel_matrix)
        k_d_tree = KDTree(positive_samples)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(
                BeeHive(k_d_tree, self.population_size, self.cycle_numbers, self.fit_neighbors,
                        self.sight, self.chosen_number, self.fit_function).optimize_positions,
                np.array(positive_samples)) for _ in range(self.generations)]
            print(f'added {len(futures)} jobs to executor')
            wait(futures, return_when='FIRST_EXCEPTION')
            print('finished ABC Algorithm')
            results = [f.result() for f in futures]
            # results = [BeeHive(k_d_tree, self.population_size, self.cycle_numbers, self.fit_neighbors,
            #         self.sight, self.chosen_number).optimize_positions(np.array(positive_samples))]
        self.create_rules(np.concatenate([[bee[1] for bee in res] for res in results]),
                          postive_test, negative_test)

    def classify(self, packet: []) -> float:
        """
        Classifies the given packet with the trained model
        The closest cluster is taken into account and if the sample is closer than twice the radius, the membership degree is
        higher than 0
        :param packet: the normalized packet to classify
        :return: the membership degree in [0,1] where 0 is definitely not in class and 1 is definitely in class
        """
        if self.method == 'non-linear':
            packet = rbf_kernel([np.array(packet)], self.kernel_matrix)
        d, i = self.classification_tree.query(np.array(packet))
        if isinstance(i, ndarray):
            i = i[0]
        if isinstance(d, ndarray):
            d = d[0]
        r = self.radii[i - 1]
        if d >= 2 * r:
            return 0
        else:
            return 1 - (0.5 / r) * d

    def save(self, path: str) -> bool:
        """
        Saves the model
        :param path: path where the model should be saved
        :return true if the model could be saved
        """
        with open(path, 'w') as file:
            file.write(f'{self.method}\n')
            file.write(';'.join([','.join([str(s) for s in arr]) for arr in self.classification_tree.data]))
            file.write('\n')
            file.write(','.join([str(r) for r in self.radii]))
            file.write('\n')
            if 'non-linear' in self.method:
                file.write(f'{",".join([" ".join([str(value) for value in row]) for row in self.kernel_matrix])}')
            return True

    def load(self, path: str) -> bool:
        """
        loads the model
        :param path: path to load the model from
        :return: true if the model could be loaded
        """
        if not isfile(path):
            return False
        with open(path) as file:
            lines = file.readlines()
            self.method = lines[0].strip()
            arrays = lines[1].strip().split(';')
            rows = [[float(number) for number in row.split(',')] for row in arrays]
            self.classification_tree = KDTree(np.array(rows))
            self.radii = [float(i) for i in lines[2].strip().split(',')]
            if self.method == 'non-linear':
                rows = [np.fromstring(r, dtype=float, sep=' ') for r in
                        lines[3].removeprefix('[').removesuffix(']').split(',')]
                self.kernel_matrix = np.array(rows)
            return True

    def create_rules(self, bees: ndarray, positive_test_samples, negative_test_samples):
        cluster_centers = list(range(2, max(4, int(len(bees) / 10))))
        print('start FCM')
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(create_fcm, center, bees.copy()) for center in cluster_centers]
        print(f'added {len(cluster_centers)} jobs to executor')
        wait(futures, return_when='FIRST_EXCEPTION')
        fcms = [f.result() for f in futures]
        best = find_best_fcm(fcms, bees, self.method, positive_test_samples, negative_test_samples, self.kernel_matrix)
        self.classification_tree = KDTree(best.centers)
        dist = [self.classification_tree.query(p) for p in bees]
        radii = [0 for _ in range(len(best.centers))]
        for d in dist:
            if radii[d[1] - 1] < d[0]:
                radii[d[1] - 1] = d[0]
        self.radii = radii
