class Classifier:
    """
    Abstract Class of the classifiers
    """

    def train(self, data_set: [], positive_labels: [int]) -> None:
        """
        Method should train the classifier with the given data
        :param positive_labels: the labels which represent the searched label
        :param data_set: set of normalized data with only the important features present
        """
        pass

    def classify(self, packet: []) -> float:
        """
        :param packet: the normalized packet to classify
        :return: the membership degree in [0,1] where 0 is definitely not in class and 1 is definitely in class
        """
        pass

    def save(self, path: str) -> bool:
        """
        Saves the model
        :param path: path where the model should be saved
        :return true if the model could be saved
        """

    def load(self, path: str) -> bool:
        """
        loads the model
        :param path: path to load the model from
        :return: true if the model could be loaded
        """