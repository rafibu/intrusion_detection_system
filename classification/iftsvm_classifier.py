import math
from os.path import isfile
from random import randint

import numpy as np
from numpy import ndarray
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel

from classification.classifier import Classifier


def split_samples(data_set: [], positive_labels: [int]) -> (ndarray, ndarray):
    """
    splits the dataset into two distinct sets, one containing all positive and one containing all negative samples
    The last feature which contains those labels will be omitted
    :param data_set: all samples, where the last dimension contains the class labels
    :param positive_labels: all values which represent a positive class label
    :return: two ndarrays, the first containing all positive samples, the second all negative ones
    """
    positive_indices = []
    negative_indices = []
    for i in range(len(data_set)):
        if data_set[i][-1] in positive_labels:
            positive_indices.append(i)
        else:
            negative_indices.append(i)
    positive = [data_set[i] for i in positive_indices]
    negative = [data_set[i] for i in negative_indices]
    return np.delete(positive, -1, 1), np.delete(negative, -1, 1)


def calculate_center(samples: ndarray) -> (ndarray, float):
    """
    Calculates the center of all given samples and the radius
    :param samples: all samples which have the same class
    :return: the class center and the radius
    """
    center = np.sum(samples, axis=0) / len(samples)
    radius = max([math.dist(center, point) for point in samples])
    return center, radius


class Hyperplane:
    b: float
    w: ndarray

    def __init__(self, w=None, b=None):
        self.w = w
        self.b = b

    def distance(self, x):
        return abs(np.dot(np.transpose(self.w), x) + self.b) / np.linalg.norm(self.w)

    def save(self) -> str:
        return f'{self.b};; [{" ".join([str(p) for p in self.w])}]'

    def load(self, string: str):
        attributes = [a.strip() for a in string.split(';;')]
        self.b = float(attributes[0])
        self.w = np.fromstring(attributes[1].removeprefix('[').removesuffix(']'), dtype=float, sep=' ')
        return self


def get_random_elements(matrix, size) -> ndarray:
    indices = []
    for _ in range(size):
        rnd = randint(0, len(matrix) - 1)
        while rnd in indices:
            rnd = randint(0, len(matrix) - 1)
        indices.append(rnd)
    return matrix[indices]


def create_kernel_matrix(A: ndarray, B: ndarray, kernel_size: int) -> ndarray:
    if kernel_size is None:
        kernel_size = int((len(A) + len(B)) / 100)
    elif kernel_size / 2 > len(A) or kernel_size / 2 > len(B):
        raise RuntimeError(f'Kernel size {kernel_size} too large for matricies with lengths: {len(A)}, {len(B)}')
    kernel_a = get_random_elements(A, int(kernel_size / 2))
    kernel_b = get_random_elements(B, int(kernel_size - len(kernel_a)))
    return np.concatenate((kernel_a, kernel_b), axis=0)


class IFTSVMClassifier(Classifier):
    """
    This class represents the Intuitionistic Fuzzy Twin Support Vector Machine for classification
    This SVM uses two hyperplanes instead of just one. One for membership and the other for non-membership
    The SVM can either use linear classification or non-linear classification, where the data uses an RBF Kernel
    """

    positive_hyperplane: Hyperplane
    negative_hyperplane: Hyperplane

    dual_mode = False

    def __init__(self, method='linear', delta=10, alpha=30, C_1=1, C_2=1, C_3=1, C_4=1, kernel_size=None):
        """
        :param method: either 'linear' or 'non-linear' (use a rbf-kernel)
        :param delta: parameter used for membership function which describes the fuzziness of the class-radius
        :param alpha: radius for non-membership function, counts all neighbors within that radius to create score fct
        :param C_1: penalty parameter for hyperplane generation
        :param C_2: penalty parameter for hyperplane generation
        :param C_3: penalty parameter for hyperplane generation
        :param C_4: penalty parameter for hyperplane generation
        :param kernel_size: only used with non-linear function, defines how big the rbf-kernel should be,
                            default is 1% of training data set size. Big numbers slow down the classification process
        """
        self.kernel_size = kernel_size
        self.delta = delta
        self.alpha = alpha
        self.C_1 = C_1
        self.C_2 = C_2
        self.C_3 = C_3
        self.C_4 = C_4
        if method not in ['linear', 'non-linear']:
            raise BaseException('method should be either linear of non-linear')
        self.method = method
        self.kernel_matrix = None

    def train(self, data_set: [], positive_labels: [int]):
        """
        Trains the model and tries to approximate the two hyperplanes through the following steps:
            0. Split samples into positive and negative
            1. Calculate class center of positive & negative samples
            2. calculate radius of classes
            3. Calculate Intuitionistic Fuzzy Membership (IFN) for each training sample (membership and non-membership)
            4. Calculate score function for each sample -> filter out outliers
            5. calculate membership and non-membership hyperplanes
        :param positive_labels: the labels which represent the searched label
        :param data_set: set of normalized data with only the important features present
        """
        np.random.shuffle(data_set)
        positive_samples, negative_samples = split_samples(data_set, positive_labels)
        positive_class_center, positive_radius = calculate_center(positive_samples)
        negative_class_center, negative_radius = calculate_center(negative_samples)
        positive_samples = self.calculate_membership(positive_samples, positive_class_center, positive_radius)
        negative_samples = self.calculate_membership(negative_samples, negative_class_center, negative_radius)
        # Change alpha to accommodate a more dynamically accurate measurement
        self.alpha = positive_radius / 100
        positive_samples = self.calculate_non_membership_and_score(positive_samples, negative_samples)
        self.alpha = negative_radius / 100
        negative_samples = self.calculate_non_membership_and_score(negative_samples, positive_samples)
        self.calculate_hyperplanes(positive_samples, negative_samples)

    def classify(self, packet: []) -> float:
        """
        Classifies the given packet with the trained model
        :param packet: the normalized packet to classify
        :return: the membership degree in [0,1] where 0 is definitely not in class and 1 is definitely in class
        """
        if self.method == 'non-linear':
            packet = np.transpose(rbf_kernel(np.array([packet]), self.kernel_matrix))
        in_label = self.positive_hyperplane.distance(packet)
        non_label = self.negative_hyperplane.distance(packet)
        if in_label == 0:
            return 1
        if non_label == 0:
            return 0
        if in_label < non_label:
            return 1 - 0.5 * in_label / non_label
        else:
            return 0.5 * non_label / in_label

    def save(self, path: str) -> bool:
        """
        Saves the model
        :param path: path where the model should be saved
        :return true if the model could be saved
        """
        with open(path, 'w') as file:
            file.write(f'{self.method}\n')
            file.writelines(
                [hyperplane.save() + '\n' for hyperplane in [self.positive_hyperplane, self.negative_hyperplane]])
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
            if self.method not in ['linear', 'non-linear']:
                print(f'Could not read method {self.method}')
                return False
            self.positive_hyperplane = Hyperplane().load(lines[1])
            self.negative_hyperplane = Hyperplane().load(lines[2])
            if self.method == 'non-linear':
                rows = [np.fromstring(r, dtype=float, sep=' ') for r in
                        lines[3].removeprefix('[').removesuffix(']').split(',')]
                self.kernel_matrix = np.array(rows)
            return True

    def membership(self, sample: ndarray, class_center: ndarray, radius: float) -> float:
        """
        calculates the membership degree of a single sample with the following formula:
            μ(x) = 1 − ||φ(x) − C+||/(r+δ)
        :param sample: sample to calculate membership degree
        :param class_center: center of the class samples
        :param radius: radius of class samples
        :return: the membership degree
        """
        if radius == 0 and self.delta == 0:
            radius = 0.1
        return 1 - math.dist(sample, class_center) / (radius + self.delta)

    def calculate_membership(self, samples: ndarray, class_center: ndarray, radius: float) -> []:
        """
        Calculates the membership degree for each sample
        :param samples: samples as an array of arrays
        :param class_center: the calculated class center
        :param radius: radius of the class
        :return: an array of tuples, where the first element is the sample and the second its membership degree
        """
        return [(s, self.membership(s, class_center, radius)) for s in samples]

    def calculate_non_membership_and_score(self, samples: [tuple], non_samples: [tuple]) -> [tuple]:
        """
        Adds the non-membership degree and the score as an added value to the tuple
        The score defines if a sample is an outlier or part of the support vector
        :param samples: samples which should be calculated
        :param non_samples: all samples which have another label
        :return: the samples appended with the non-membership degree and the score
        """
        res = []
        for sample in samples:
            mu = sample[1]
            close_non_samples = [s for s in non_samples if math.dist(sample[0], s[0]) < self.alpha]
            close_samples = [s for s in samples if (s is not sample and math.dist(sample[0], s[0]) < self.alpha)]
            if len(close_non_samples) != 0:
                nu = (1 - mu) * (len(close_non_samples) / (len(close_samples) + len(close_non_samples)))
            else:
                nu = 0
            if nu == 0:
                score = mu
            elif mu <= nu:
                score = 0
            else:
                score = (1 - nu) / (2 - mu - nu)
            res.append((sample[0], sample[1], nu, score))
        return res

    # Methods to calculate Hyperplanes

    def calculate_hyperplanes(self, positive_samples, negative_samples):
        """
        Calculates the two hyperplanes which are used to classify the samples
        :param positive_samples: all positive samples in the training data
        :param negative_samples: all negative samples in the training data
        :return: the two calculated hyperplanes
        """
        A = np.array([s[0] for s in positive_samples])
        B = np.array([s[0] for s in negative_samples])
        if self.method == 'non-linear':
            self.kernel_matrix = create_kernel_matrix(A, B, self.kernel_size)
            A = rbf_kernel(A, self.kernel_matrix)
            B = rbf_kernel(B, self.kernel_matrix)
        e_1 = np.ones(len(positive_samples))
        e_2 = np.ones(len(negative_samples))
        s_1 = np.array([s[-1] for s in positive_samples])
        s_2 = np.array([s[-1] for s in negative_samples])

        x_start = np.ones(len(A[0]) + 1 + len(B))
        result = minimize(self.min_A, x_start, method='trust-constr',
                          constraints={'type': 'eq', 'fun': self.constraints_A, 'args': (A, B, e_2)},
                          args=(A, e_1, s_2))
        x = result.get('x')
        w_1 = np.array([x[i] for i in range(len(A[0]))])
        b_1 = x[len(A[0])]
        self.positive_hyperplane = Hyperplane(w=w_1, b=b_1)

        x_start = np.ones(len(B[0]) + 1 + len(A))
        result = minimize(self.min_B, x_start, method='trust-constr',
                          constraints={'type': 'eq', 'fun': self.constraints_B, 'args': (A, B, e_1)},
                          args=(B, e_2, s_1))
        x = result.get('x')
        w_2 = np.array([x[i] for i in range(len(B[0]))])
        b_2 = x[len(B[0])]
        self.negative_hyperplane = Hyperplane(w=w_2, b=b_2)

        return self.positive_hyperplane, self.negative_hyperplane

    def min_A(self, x, A, e_1, s_2):
        """
        method which calculates the to minimize formula
        min(w1 ,b1 ,ξ2){1/2 ||A*w1 + e1*b1||^2 + 1/2*C1 * ||w1||^2 + C2 * s2^T * ξ2}
        :param x: all variables used to optimize the hyperplane (w_1, b_1 and xi_2)
        :return: a scalar which should be minimized by changing the input x
        """
        w_1 = np.array([x[i] for i in range(len(A[0]))])
        b_1 = x[len(A[0])]
        xi_2 = np.array([x[i] for i in range(len(A[0]) + 1, len(x))])
        return 0.5 * np.linalg.norm(np.dot(A, w_1) + e_1 * b_1) ** 2 + 0.5 * self.C_1 * np.linalg.norm(
            w_1) ** 2 + self.C_2 * np.dot(np.transpose(s_2), xi_2)

    def min_B(self, x, B, e_2, s_1):
        """
        method which calculates the to minimize formula
        min(w2 ,b2 ,ξ1){1/2 ||B*w2 + e2*b2||^2 + 1/2*C3 * ||w2||^2 + C4 * s1^T * ξ1}
        :param x: all variables used to optimize the hyperplane (w_1, b_1 and xi_2)
        :return: a scalar which should be minimized by changing the input x
        """
        w_2 = np.array([x[i] for i in range(len(B[0]))])
        b_2 = x[len(B[0])]
        xi_1 = np.array([x[i] for i in range(len(B[0]) + 1, len(x))])
        return 0.5 * np.linalg.norm(
            np.dot(B, w_2) + e_2 * b_2) ** 2 + 0.5 * self.C_3 * np.linalg.norm(
            w_2) ** 2 + self.C_4 * np.dot(np.transpose(s_1), xi_1)

    def constraints_A(self, x, A, B, e_2):
        """
        This method checks the constraints when minimizing the positive samples
        The two constraints are:
        1. -(B*w_1 + e_2*b_1) + ξ_2 ≥ e_2
        2.  ξ_2 ≥ 0
        :param x: all variables used to optimize the hyperplane (w_1, b_1 and xi_2)
        :return: the number of constraint violations with the current solution
        """
        w_1 = np.array([x[i] for i in range(len(A[0]))])
        b_1 = x[len(A[0])]
        xi_2 = np.array([x[i] for i in range(len(A[0]) + 1, len(x))])
        const = -(np.dot(B, w_1) + e_2 * b_1) + xi_2 - e_2
        return len([c for c in const if c < 0]) + len([s for s in xi_2 if s < 0])

    def constraints_B(self, x, A, B, e_1):
        """
        This method checks the constraints when minimizing the negative samples
        The two constraints are:
        1. (A*w_2 + e_1*b_2) + ξ_1 ≥ e_1
        2.  ξ_1 ≥ 0
        :param x: all variables used to optimize the hyperplane (w_2, b_2 and xi_1)
        :return: the number of constraint violations with the current solution
        """
        w_2 = np.array([x[i] for i in range(len(B[0]))])
        b_2 = x[len(B[0])]
        xi_1 = np.array([x[i] for i in range(len(B[0]) + 1, len(x))])
        const = np.dot(A, w_2) + e_1 * b_2 + xi_1 - e_1
        return len([c for c in const if c < 0]) + len([s for s in xi_1 if s < 0])
