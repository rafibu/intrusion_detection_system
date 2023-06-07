import time
from os.path import isfile
from random import randint, random

import numpy as np
from numpy import ndarray
from sklearn.neighbors import NearestNeighbors

from feature_selection.correlation_selector import change_attack_label, load
from feature_selection.feature_selector import FeatureSelector


def reduce_features(sample: [], features: [int]) -> []:
    """
    method reduces the sample to the selected features for further processing
    :param sample: the sample of normalized data
    :param features: the features which should be active at the end
    :return: the sample data with the given features
    """
    to_delete = [i for i in range(len(sample)) if i not in features]
    return np.delete(sample, to_delete)


def reduce_bulk_features(data_set: [[]], features: [int]):
    """
    reduces the features of a whole dataset
    :param data_set: all data which should be reduced
    :param features: features which should be kept, starting at 0
    :return: the copy of the dataset with only the features given
    """
    to_delete = [i for i in range(len(data_set[0])) if i not in features]
    return np.delete(data_set, to_delete, axis=1)


def create_random_population(chr_number: int, population_size: int) -> [[int]]:
    """
    Creates chromosomes with random activated alleles
    :param chr_number: number of alleles per chromosome
    :param population_size: number of chromosomes
    :return: an array of ndarrays which represent the chromosomes
    """
    result = []
    for _ in range(population_size):
        result.append([randint(0, 1) for _ in range(chr_number)])
    return result


def calculate_fitness_function(chromosome: [int], data_set: ndarray, k: int) -> float:
    """
    This function removes all features which aren't active in the chromosome (i.e. have the value 0) and then calculates
    the k-nearest neighbors in the validation_set for each of the samples in the set. The fitness function is then
    calculated by comparing how many of these nearest neighbors have the same label
    :param k: the number of nearest neighbors
    :param chromosome: the chromosome which should be evaluated
    :param data_set: the samples which are used to calculate the fitness function
    :return: number in [0,1] depending on how many of the samples are labeled like their neighbors
    """
    classes = np.delete(data_set, range(len(data_set[0]) - 1), 1)
    rows_to_delete = []
    for i in range(len(chromosome)):
        if chromosome[i] == 0:
            rows_to_delete.append(i)
    # remove class labels
    rows_to_delete.append(len(data_set[0]) - 1)
    evaluation_set = np.delete(data_set, rows_to_delete, 1)
    neighbors = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(evaluation_set)
    indices = neighbors.kneighbors(evaluation_set, return_distance=False)
    # remove the closest elements as they are themselves
    indices = np.delete(indices, 0, 1)
    correct = []
    for i in range(len(indices)):
        clazz = classes[i]
        correct.append(sum([1 for c in indices[i] if classes[c] == clazz]))
    fitness = sum(correct) / (k * len(correct))
    return fitness


def order_by_fitness(population: [[int]], data_set: ndarray, k: int) -> None:
    population.sort(key=lambda i: calculate_fitness_function(i, data_set, k), reverse=True)


def cross_over(chr1: [int], chr2: [int]) -> [int]:
    crossover = []
    for i in range(len(chr1)):
        if random() > 0.5:
            crossover.append(chr1[i])
        else:
            crossover.append(chr2[i])
    return crossover


def calculate_feature_list(best_chromosomes: [[int]]) -> [tuple]:
    """
    Calculates how frequent each feature is represented in the best chromosomes
    :param best_chromosomes: list of the best chromosomes
    :return: array of tuples, where the first element is the index of a feature and the second is the frequency
    """
    result = []
    for i in range(len(best_chromosomes[0])):
        result.append((i, sum([j[i] for j in best_chromosomes]) / len(best_chromosomes)))
    result.sort(key=lambda f: f[1], reverse=True)
    return result


class GenAlgSelector(FeatureSelector):
    """
    A feature selection algorithm which utilizes a genetic algorithm to find the most important
    features by training with labeled data
    """

    calculated_feature_list = []

    def __init__(self, initial_population_size=100, mutation_rate=0.02, elite_children=0.2, generations=30,
                 best_to_retain=0.2, k=5):
        """
        :param initial_population_size: 
        :param mutation_rate: chance of a chromosome to change randomly
        :param elite_children: fraction of children which are elite children and will not change for the next generation
        :param generations: Number of generations in which the chromosomes evolve
        :param best_to_retain: fraction of chromosomes which should be kept at the end of the evolution
        :param k: number for the k nearest neighbor algorithm used in the fitness function
        """
        self.initial_population_size = initial_population_size
        self.mutation_rate = mutation_rate
        self.elite_children = elite_children
        self.generations = generations
        self.best_to_retain = best_to_retain
        self.k = k

    def train(self, data: [], train_labels: [int]) -> [int]:
        """
        this method uses the data and the corresponding labels to train the GA. A 5-fold Cross Validation is used
        The GA uses a fitness function which evaluates if a given chromosome creates a feasible result. This fitness
        function uses KNN, i.e. for each sample it looks at the K-nearest neighbors according to the active features
        and checks how many of their labels are the same.
        The following steps describe the whole training process:
            1. Split the whole training set into 5 distinct subsets for CV
            2. for each subset S
                a. Create a random population of chromosomes
                b. repeat for each generation:
                    i. check fitness for each
                    ii. remove lower half and double upper half
                    iii. Evolve non-elite chromosomes (Mutation and Crossover)
                c. save the best chromosomes as tuples with fitness
            3. Order features by counting how often they are active in the saved chromosomes
        :param data: the normalized training data
        :param train_labels: the labels which correspond to an attack
        :return: an ordered array of the features
        """
        chr_number = len(data[0])
        if len(train_labels) > 1:
            # change all attack labels for binary classification, since this might create higher correlation
            data = [change_attack_label(d, train_labels) for d in data]
        if not isinstance(data, ndarray):
            data = np.array(data)
        np.random.shuffle(data)
        splits = 5
        data_arrays = np.array_split(data, splits)
        best_chromosomes = []
        for v in range(splits):
            start = time.time()
            validation_set = data_arrays[v].copy()
            training_set = np.concatenate([d for d in data_arrays if d is not validation_set])
            population = create_random_population(chr_number, self.initial_population_size)
            for _ in range(self.generations):
                order_by_fitness(population, training_set, self.k)
                population = self.evolve(population)
            order_by_fitness(population, validation_set, self.k)
            best_chromosomes.extend(self.get_best(population))
            print(f'{v}. set calculated (time {round(time.time() - start)}s')
        self.calculated_feature_list = calculate_feature_list(best_chromosomes)
        return [feat[0] for feat in self.calculated_feature_list]

    def get_with_threshold(self, threshold: float) -> [int]:
        """
        This method is used, when all features should be returned which have a membership degree above the threshold
        :param threshold: the threshold in [0,1]
        :return: a list of all features which are above the threshold
        """
        return [f[0] for f in self.calculated_feature_list if f[1] >= threshold]

    def get_thresholds(self):
        """
        :return: ordered list of all scores of the features
        """
        copy = self.calculated_feature_list.copy()
        copy.sort(key=lambda a: a[0])
        return [a[1] for a in copy]

    def get_highest_ranked_features(self, number_of_features: int) -> [int]:
        """
        This method is used, when a certain number of features should be chosen, which have the highest membership degree
        :param number_of_features:
        :return: a list of n features, which have the highest membership degree
        """
        return [t[0] for t in self.calculated_feature_list[:number_of_features]]

    def evolve(self, population: [[int]]) -> [[int]]:
        """
        Preserves the best members of the population and evolves the others through cross over and random mutation
        :param population: chromosomes represented as arrays of 0 and 1
        :return: a new population the same size as the old one
        """
        size = len(population)
        # Remove bottom half
        population = [population[i] for i in range(round(len(population) / 2))]
        # Leave the best fitting population unchanged
        new_population = [population[i] for i in range(round(len(population) * self.elite_children))]
        half_size = len(population) - 1
        while len(new_population) < size:
            first = randint(0, half_size)
            second = randint(0, half_size)
            while second == first:
                second = randint(0, half_size)
            crossover = cross_over(population[first], population[second])
            for i in range(len(crossover)):
                # random mutations
                if random() <= self.mutation_rate:
                    crossover[i] = (crossover[i] + 1) % 2
            new_population.append(crossover)
        return new_population

    def get_best(self, population):
        return [population[i] for i in range(round(len(population) * self.best_to_retain))]

    def load_selected_features(self, file_path: str) -> bool:
        """
        loads the calculated features from a file
        :param file_path: the path to the file to read and parse
        :return: true if the file could be parsed successfully
        """
        if not isfile(file_path):
            return False
        with open(file_path) as file:
            self.calculated_feature_list = [load(line) for line in file]
            # Since the other methods depend on a sorted list, we sort it here again
            self.calculated_feature_list.sort(key=lambda f: f[1], reverse=True)
            return True

    def save_selected_features(self, file_path: str) -> bool:
        """
        saves the features in a persistent file
        :param file_path: The path to save the file
        :return: true if saving was successful
        """
        with open(file_path, 'w') as file:
            file.writelines([f'{t[0]}, {t[1]}' + '\n' for t in self.calculated_feature_list])
            return True
