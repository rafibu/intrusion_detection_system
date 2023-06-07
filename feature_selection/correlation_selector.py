import pandas as pd

from feature_selection.feature_selector import FeatureSelector


def change_attack_label(sample: [], attack_labels: [int]) -> []:
    """
    changes the last label to the first attack label, if it is in the list of attack labels
    :param sample: the sample to change
    :param attack_labels: all labels which are attack labels
    :return: the possibly changed sample
    """
    if sample[-1] in attack_labels:
        sample[-1] = attack_labels[0]
    return sample


def load(line) -> tuple:
    elements = line.split(',')
    return int(elements[0].strip()), float(elements[1].strip())


class CorrelationSelector(FeatureSelector):
    """
    A feature selection algorithm which utilizes simple correlation between the features and the attack labels
    """
    # tuples with the feature index and the correlation
    feature_corr_tuples = [tuple]

    def __init__(self, method='pearson'):
        """
        :param method: correlation method to use
        """
        self.method = method

    def train(self, data: [], train_labels: [int]) -> [int]:
        """
        this method uses the data and the corresponding labels to find the ones with the highest correlation
        the last feature has to be the one carrying the attack/normal label
        :param data: the normalized training data
        :param train_labels: the labels which correspond to an attack
        :return: an array, with the correlation for each feature
        """
        if len(train_labels) > 1:
            # change all attack labels for binary classification, since this might create higher correlation
            data = [change_attack_label(d, train_labels) for d in data]
        data = [tuple(d) for d in data]
        data_frame = pd.DataFrame(data, columns=range(len(data[0])))
        correlation = data_frame.corr(method=self.method)
        tuples = []
        for i in range(len(correlation) - 1):
            tuples.append((i, abs(correlation[len(correlation) - 1][i])))
        self.feature_corr_tuples = tuples
        return [abs(s[1]) for s in tuples]

    def get_with_threshold(self, threshold: float) -> [int]:
        """
        This method is used, when all features should be returned which have a membership degree above the threshold
        :param threshold: the threshold between 0 and 1
        :return: a list of all features which are above the threshold
        """
        # Maybe remove if inter-correlation too big
        return [i for (i, corr) in self.feature_corr_tuples if corr >= threshold]

    def get_thresholds(self):
        """
        :return: ordered list of all scores of the features
        """
        copy = self.feature_corr_tuples.copy()
        copy.sort(key=lambda a: a[0])
        return [a[1] for a in copy]

    def get_highest_ranked_features(self, number_of_features: int) -> [int]:
        """
        This method is used, when a certain number of features should be chosen, which have the highest membership degree
        :param number_of_features:
        :return: a list of n features, which have the highest membership degree
        """
        self.feature_corr_tuples.sort(key=lambda a: -abs(a[1]))
        # Maybe remove if inter-correlation too big
        return [t[0] for t in self.feature_corr_tuples[:number_of_features]]
