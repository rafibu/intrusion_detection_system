class FeatureSelector:
    """
    Abstract Class for Feature Selectors
    """

    def train(self, data: [], train_labels: [int]) -> [int]:
        """
        This method should train the Feature Selector and create a ranked list of the features
        :param data: training data to create feature selection
        :param train_labels: the labels which define a positive sample (e.g. an attack)
        :return: the ranked list of the features
        """
        pass

    def get_with_thresholds(self, threshold: float) -> [int]:
        """
        Get all features which are over a certain threshold
        :param threshold: the threshold in [0,1] which a features correlation must be higher than
        :return: all features passing the threshold
        """
        pass

    def get_highest_ranked_features(self, number_of_features: int) -> [int]:
        """
        Get the number of the features with the highest membership degree
        :param number_of_features: The number of features which should be taken into account
        :return: a list of the numbers of the features
        """
        pass
