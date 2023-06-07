import time
from copy import deepcopy

import numpy as np
from numpy import ndarray

from classification.abc_classifier import ABCClassifier
from classification.classifier import Classifier
from classification.iftsvm_classifier import IFTSVMClassifier
from feature_selection.correlation_selector import CorrelationSelector
from feature_selection.feature_selector import FeatureSelector
from feature_selection.genetic_algorithm_selector import reduce_bulk_features
from training_process.trained_model import TrainedModel


def log(s: str):
    """
    Simple log function which adds the current time to a print statement
    """
    print(f'{time.strftime("%d.%m.%Y %H:%M:%S", time.localtime())}, {s}')


def validate_classifier(classifier, test_data):
    """
    Splits the test data into four arrays, which can be used to calculate the measures. The validation process uses a
    threshold of 0.5
    :return: The test data in four arrays true_positive, false_positive, true_negative, false_negative
    """
    labels = [d[-1] for d in test_data]
    test_data = np.delete(test_data, -1, axis=1)
    if len(classifier) == 1:
        return validate_single_classifier(classifier[0], labels, test_data)
    return validate_dual_classifier(classifier, labels, test_data)


def validate_single_classifier(classifier, labels, test_data):
    """
    Validates a classifier which  is in single mode
    """
    classifications = [[classifier.classify(sample), sample] for sample in test_data]
    true_positive = []
    true_negative = []
    false_positive = []
    false_negative = []
    for i in range(len(classifications)):
        test = classifications[i][0]
        label = labels[i]
        if label != 1:
            if test >= 0.5:
                true_positive.append(classifications[i])
            else:
                false_negative.append(classifications[i])
        else:
            if test >= 0.5:
                false_positive.append(classifications[i])
            else:
                true_negative.append(classifications[i])
    return true_positive, false_positive, true_negative, false_negative


def validate_dual_classifier(classifiers, labels, test_data):
    """
    Validates a classifier in dual mode, the normal classifiers results are added to the attack classifier ones
    """
    tp, fp, tn, fn = validate_single_classifier(classifiers[0], labels, test_data)
    fn_n, tn_n, tp_n, fp_n = validate_single_classifier(classifiers[1], labels, test_data)
    tp.extend(tp_n)
    tn.extend(tn_n)
    fp.extend(fp_n)
    fn.extend(fn_n)
    return tp, fp, tn, fn


def calculate_measures(true_positive, false_positive, true_negative, false_negative):
    """
    Calculates the three measures
        - Accuracy
        - Detection Rate
        - False Alarm Rate
    From the four arrays given
    :return: Dictionary holding the three values
    """
    tp = len(true_positive)
    tn = len(true_negative)
    fp = len(false_positive)
    fn = len(false_negative)
    acc = (tp + tn) / (tp + tn + fp + fn)
    dr = tp / (tp + fn)
    far = fp / (fp + tn)
    return {'acc': acc, 'dr': dr, 'far': far}


def get_worst(samples_scores, original_data, features, fraction: float):
    """
    :return: the worst classified fraction of the original data to use in the next iteration
    """
    samples_scores.sort(key=lambda a: a[0])
    samples = [s[1] for s in samples_scores]
    original = []
    for sample in np.array_split(samples, 100 * fraction)[0]:
        for data in original_data:
            if equal(sample, data, features):
                original.append(data)
                break
    return original


def equal(sample, data, features):
    """
    Checks if the data sample is the same as the sample given the used features
    :return: True if the two samples have the same values in the given features
    """
    reduced = [data[f] for f in features]
    return all(sample == reduced)


def equalize_data(data: [], positive_labels: []) -> ndarray:
    """
    Equalizes the data set by reducing the largest of positive or negative to the same number of samples as the other
    :param data: all data
    :param positive_labels: the positive labels as an array
    :return: a ndarray containing the same amount of positive and negative samples
    """
    is_positive = [d[-1] in positive_labels for d in data]
    positive = [data[i] for i in range(len(data)) if is_positive[i]]
    negative = [data[i] for i in range(len(data)) if not is_positive[i]]
    smaller, larger = (min(positive, negative, key=lambda l: len(l)), max(positive, negative, key=lambda l: len(l)))
    smaller.extend(larger[:len(smaller)])
    return np.array(smaller)


def train_model(data: [], positive_labels: [], feature_selectors=[CorrelationSelector()],
                classifiers=[ABCClassifier(method='non-linear', kernel_size=50),
                             IFTSVMClassifier(method='non-linear', kernel_size=50)],
                iterations=5, selector_threshold=0.5,
                training_data_length=1000) -> TrainedModel:
    """
    This function trains the whole model holistically using an iterative optimization process which tries to optimize
    the training data without overfitting
    :param data: the already normalized data with all features present
    :param feature_selectors: the feature selection objects instantiated but not necessarily trained
    :param classifiers: the classifier objects instantiated but not necessarily trained
    :param iterations: Number of iterations over the whole process
    :param selector_threshold: What threshold the features have to meet to be chosen by the selection process
    :param training_data_length: Length of data split, which is used to train the classifiers in each iteration
    :return: a TrainedModel which can classify samples using the trained algorithms
    """
    log(f'Start training model with {len(feature_selectors)} feature selectors and {len(classifiers)} classifiers')
    np.random.shuffle(data)
    split_data = np.array_split(data, len(data) / training_data_length)
    split = 0
    test_data = split_data[split]
    selector_trainer = FeaturesSelectionTrainer(feature_selectors, positive_labels)
    classifier_trainer = ClassifierTrainer(classifiers)
    for i in range(iterations):
        log(f'start iteration {i}')
        features = selector_trainer.calculate_features(test_data, selector_threshold)
        # Need to append the label
        log(f'Finished training feature selectors, features: {features}')
        features.append(len(data[0]) - 1)
        reduced_data = reduce_bulk_features(test_data, features)
        classifier_trainer.train(reduced_data, positive_labels)
        samples_scores = classifier_trainer.validate_classifiers(
            reduce_bulk_features(equalize_data(np.concatenate(split_data[(split + 1):(split + 21)]), positive_labels),
                                 features))
        if i != iterations - 1:
            # selector_trainer.validate_selectors(samples_scores)
            features.remove(len(data[0]) - 1)
            wrongly_classified = get_worst(samples_scores, split_data[split], features, 0.1)
            split += 1
            test_data = np.concatenate((wrongly_classified, split_data[split]))
        else:
            features.remove(len(data[0]) - 1)
            return TrainedModel(classifiers=classifier_trainer.get_ranked_list(), features=features)


class FeaturesSelectionTrainer:

    def __init__(self, feature_selectors: [FeatureSelector], positive_labels: []):
        self.features_selectors = feature_selectors
        # Each Classifier gets a score, a high score means that the chosen features are more important
        self.scores = np.ones(len(feature_selectors))
        self.positive_labels = positive_labels

    def calculate_features(self, train_data: ndarray, threshold: float):
        """
        Calculates which features should be used by comparing their importance to the threshold given
        :param train_data: the data which the selectors are trained on
        :param threshold: the threshold which a feature has to meet to be chosen
        :return: an array of the features
        """
        for selector in self.features_selectors:
            selector.train(train_data, self.positive_labels)
        feature_scores = np.zeros(len(train_data[0]) - 1)
        for i in range(len(self.features_selectors)):
            selector = self.features_selectors[i]
            features = np.array(selector.get_thresholds())
            feature_scores += features
        chosen_features = [i for i in range(len(feature_scores)) if
                           (feature_scores[i]) / len(self.features_selectors) >= threshold]
        return chosen_features

    def validate_selectors(self, samples_scores):
        for score in samples_scores:
            if score[1][-1] in self.positive_labels:
                # Change to Misclassification Metric -> Low scores mean higher misclassification
                score[0] = 1 - score[0]
        feature_length = len(samples_scores[0][1])
        weights = np.zeros(feature_length)
        for f in range(feature_length):
            avg = sum([score[1][f] for score in samples_scores]) / len(samples_scores)
            if isinstance(samples_scores[0][0], list):
                weights[f] = sum([score[0][0] * abs(score[1][f] - avg) for score in samples_scores])
            else:
                weights[f] = sum([score[0] * abs(score[1][f] - avg) for score in samples_scores])
        scores = np.zeros(len(self.features_selectors))
        for i in range(len(self.features_selectors)):
            selector = self.features_selectors[i]
            memberships = selector.get_thresholds()
            scores[i] = sum([memberships[f] * weights[f] for f in range(feature_length)])
        norm_factor = max(scores)
        self.scores = [score / norm_factor for score in scores]
        print(f'scores for feature selectors: {self.scores}')

    def get_ranked_list(self):
        c = self.features_selectors.copy()
        self.features_selectors.sort(key=lambda a: self.scores[c.index(a)], reverse=True)
        print(f'Ranked list of feature selectors: {self.features_selectors}')
        return self.features_selectors


class ClassifierTrainer:

    def __init__(self, classifiers: [Classifier]):
        self.classifiers = []
        for classifier in classifiers:
            if classifier.dual_mode:
                self.classifiers.append([classifier, deepcopy(classifier)])
            else:
                self.classifiers.append([classifier])
        self.scores = np.ones(len(classifiers))

    def train(self, data: ndarray, positive_labels: []):
        """
        Trains all classifiers on the given data
        :param data:
        :param positive_labels:
        """
        for classifier in self.classifiers:
            log(f'Training Classifier: {classifier}')
            classifier[0].train(data, positive_labels)
            if len(classifier) == 2:
                negative_labels = [i for i in list(range(max(positive_labels) + 1)) if i not in positive_labels]
                classifier[1].train(data, negative_labels)

    def validate_classifiers(self, test_data):
        """
        validates the classifiers on the given data
        :param test_data: data to test the classifiers with
        :return: all samples with their corresponding calculated membership degree
        """
        best_samples = []
        best_score = 0
        for i in range(len(self.classifiers)):
            classifier = self.classifiers[i]
            tp, fp, tn, fn = validate_classifier(classifier, test_data)
            acc = (len(tp) + len(tn)) / (len(tp) + len(tn) + len(fp) + len(fn))
            print(f'classifier {classifier} has accuracy: {acc}')
            self.scores[i] = acc
            if acc > best_score:
                tp.extend(fp)
                tp.extend(tn)
                tp.extend(fn)
                best_samples = tp
                best_score = acc
        return best_samples

    def get_ranked_list(self):
        c = self.classifiers.copy()
        self.classifiers.sort(key=lambda a: self.scores[c.index(a)], reverse=True)
        print(f'Ranked Classifiers: {self.classifiers}')
        return self.classifiers
