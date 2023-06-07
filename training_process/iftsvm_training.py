import copy
import datetime
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from classification.iftsvm_classifier import IFTSVMClassifier


def calculate_measures(classifier, test_data):
    """
    Calculates the three measures
        - False Alarm Rate
        - Accuracy
        - Detection Rate
    For a threshold of 0.5
    :param classifier: Trained Classifier to test
    :param test_data: Data used to calculate measures
    :return: The three values as a tuple
    """
    labels = [d[-1] for d in test_data]
    test_data = np.delete(test_data, -1, axis=1)
    classifications = [classifier.classify(sample) for sample in test_data]
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    for i in range(len(classifications)):
        test = classifications[i]
        label = labels[i]
        if label != 1:
            if test >= 0.5:
                true_positive.append(test)
            else:
                false_negative.append(test)
        else:
            if test >= 0.5:
                false_positive.append(test)
            else:
                true_negative.append(test)
    tp = len(true_positive)
    tn = len(true_negative)
    fp = len(false_positive)
    fn = len(false_negative)
    acc = (tp + tn) / (tp + tn + fp + fn)
    dr = tp / (tp + fn)
    far = fp / (fp + tn)
    return far, acc, dr


def preliminary_result(a, d, k, c_1, c_2, c_3, c_4, method, dataset, positive_labels):
    """
    Trains one IFTSVM Classifier with the given hyperparameters and evaluates its effectiveness through the formula
        far - acc - dr
    The score should be minimal for maximum effectiveness
    :return: a tuple with the score, the hyperparameters and the trained model
    """
    try:
        current = IFTSVMClassifier(method=method, alpha=a, delta=d, C_1=c_1, C_2=c_2,
                                   C_3=c_3, C_4=c_4, kernel_size=k)
        split_data = np.array_split(dataset, 100)
        current.train(split_data[0], copy.deepcopy(positive_labels))
        test_data = np.concatenate(split_data[1:5])
        far, acc, dr = calculate_measures(current, test_data)
        score = far - acc - dr
    except:
        current = None
        score = 3
    return score, [a, d, c_1, c_2, c_3, c_4, k], current


class IFTSVMTrainer:

    def __init__(self, dataset: [], positive_labels: [int], method: str):
        self.dataset = dataset
        self.positive_labels = positive_labels
        self.method = method
        self.counter = 0

    def find_best_coefficients(self, path=None):
        """
        This method tries to find the best coefficients for the following measures:
            1. False Alarm Rate
            2. Accuracy
            3. Detection Rate
        The following Coefficients will be optimized:
            1. alpha, delta (Membership degree function)
            2. C_1 - C_4 (Penalty Parameters of the hyperplane optimization)
            3. kernel size (if the method is non-linear)
        The method uses small batches of data as training sets to find an optimal solution and then checks,
        if the solution holds for bigger
        training sets.
        :param path: path where the optimized, trained classifier should be saved
        """
        start = time.time()
        print('start training')
        np.random.shuffle(self.dataset)
        alpha = [10, 50, 100]
        delta = [0, 5, 10, 100]
        C = [0.1, 0.5, 1]
        kernel_size = [5, 10, 50, 100]
        classifier = self.approximate_best(alpha, delta, C, kernel_size)
        if path:
            classifier.save(path)
        data = np.array_split(self.dataset, 100)
        far, acc, dr = calculate_measures(classifier, np.concatenate(data[1:10]))
        print(f'FAR: {far}, Accuracy: {acc}, Detection Rate: {dr}')
        print(f' time used: {datetime.timedelta(seconds=int(time.time() - start))}')
        return classifier

    def approximate_best(self, alpha, delta, C, kernel_size) -> IFTSVMClassifier:
        """
        Tests the given combinations and evaluates their success by taking small slices of the data set and training
        classifiers with them
        :return: the best Classifier
        """
        futures = []
        jobs = 0
        with ProcessPoolExecutor() as executor:
            for a in alpha:
                for d in delta:
                    for k in kernel_size:
                        for c_1 in C:
                            for c_2 in C:
                                for c_3 in C:
                                    for c_4 in C:
                                        futures.append(
                                            executor.submit(preliminary_result, a, d, k, c_1, c_2, c_3, c_4,
                                                            self.method, self.dataset, self.positive_labels))
                                        jobs += 1
            print(f'added {jobs} jobs to executor')
            old_queue = len([f.done() for f in futures if not f.done()])
            while not all([f.done() for f in futures]) or any([f.exception() is not None for f in futures]):
                queue = len([f.done() for f in futures if not f.done()])
                if old_queue - queue > 100:
                    print(f'jobs in queue: {queue}')
                    old_queue = queue
                time.sleep(30)
            results = [future.result() for future in futures]
        results.sort(key=lambda s: s[0])
        print(f'Best Preliminary results: {[str(result[0]) + ", " + str(result[1]) for result in results[0:9]]}')
        return results[0][2]
