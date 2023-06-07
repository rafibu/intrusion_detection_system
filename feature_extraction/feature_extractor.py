import re
from os.path import isfile

from fitter import Fitter

from feature_extraction.extraction_methods import *


def is_mostly_numeric(data: [str]) -> bool:
    """
    We have to remove . and - because isnumeric() does not recognize them as numeric
    :param data: data to check
    :return: true if more than 80% of the data is numeric
    """
    numeric = [d for d in [d.replace('.', '') for d in data] if d.removeprefix('-').isnumeric()]
    return len(numeric) > len(data) * 0.8


def is_ipv4(data):
    """
    :param data: data to check
    :return: true if more than 80% of the data is in an IPv4 Format
    """
    pattern = re.compile("\\d+\\.\\d+\\.\\d+\\.\\d+")
    ipv4 = [d for d in data if pattern.match(d) is not None]
    return len(ipv4) > len(data) * 0.8


def is_boolean_data(data):
    zeros = [d for d in data if float(d) == 0]
    ones = [d for d in data if float(d) == 1]
    return len(zeros) + len(ones) == len(data)


def calculate_best_numeric_normalization(numeric_data) -> ExtractionMethod:
    """
    Finds the best numeric normalization method and initializes it
    :param numeric_data: data in numeric form
    :return: the initialized extraction method
    """
    # timeout needs to be higher, since otherwise the powerlaw will not be fitted
    fitter = Fitter(numeric_data, distributions=[
        'norm',
        'powerlaw',
        'uniform'
    ], timeout=900)
    fitter.fit()
    best_dist = fitter.get_best()
    print(best_dist)
    match list(best_dist.keys())[0]:
        case 'norm':
            print('Z-Scale')
            return ZScale(numeric_data)
        case 'powerlaw':
            print('Log-Scale')
            return LogScaling()
        case 'uniform':
            print('Scale-to-Range')
            return ScaleToRange(numeric_data)


class FeatureExtractor:
    """
    This class creates a feature extractor which can automatically calculate the optimal feature extraction methods
    when given an array of training samples.
    """
    # array of the extraction methods which should be used for the current data model
    EXTRACTION_METHODS: [ExtractionMethod]

    def __init__(self):
        self.EXTRACTION_METHODS = []

    def calculate_extraction_methods(self, training_samples: []) -> [ExtractionMethod]:
        """
        This method automatically calculates which extraction method should be used for which feature of a sample
        :param training_samples: the training samples as an array of arrays
        :return: the extraction methods as an array
        """
        for i in range(len(training_samples[0])):
            self.EXTRACTION_METHODS.append(self.__calculate_for_one_feature([data[i] for data in training_samples]))
        return self.EXTRACTION_METHODS

    def __calculate_for_one_feature(self, feature_data: [str]) -> ExtractionMethod:
        """
        calculate and initialize the best extraction method for one feature
        :param feature_data: the data from one feature as an array of strings
        :return: the optimal extraction method
        """
        try:
            if is_ipv4(feature_data):
                print('IPv4')
                return IPv4Encoding(feature_data)
            if not is_mostly_numeric(feature_data):
                print('OHE')
                return OneHotEncoding(data=feature_data)
            if is_boolean_data(feature_data):
                print('BOOL')
                return BooleanEncoding()
            numeric_data = [float(data) for data in feature_data if data.isnumeric()]
            best = calculate_best_numeric_normalization(numeric_data)
            return best
        except ValueError:
            # if an error occurs, the boolean encoding is used, since it does not transform data
            print('BOOL Error')
            return BooleanEncoding()

    def change_method(self, number: int, method: ExtractionMethod) -> ExtractionMethod:
        """
        Changes the extraction method of a single feature manually
        :param number: the place of the feature starting with 0
        :param method: the new method which should be used
        :return: the new extraction method
        """
        self.EXTRACTION_METHODS[number] = method
        return method

    def trim(self, features_to_remove: [int]) -> int:
        """
        removes the features from the saved model. This method should be used after the feature selection process
        selected the appropriate features which are needed
        :param features_to_remove: an array of ints which define which extraction methods should be removed from the
        model
        :return: the number of features which have been removed
        """
        count = 0
        for number in features_to_remove:
            if self.EXTRACTION_METHODS[number] is not None:
                self.EXTRACTION_METHODS[number] = None
                count += 1
        self.EXTRACTION_METHODS = [i for i in self.EXTRACTION_METHODS if i is not None]
        return count

    def transform(self, sample: [str]) -> []:
        """
        Transforms a sample according to the current extraction methods model which has to be calculated or loaded
        beforehand
        :param sample: the sample to transform
        :return: the transformed and normalized sample
        """
        return [self.EXTRACTION_METHODS[i].transform(sample[i]) for i in range(len(sample))]

    def load_extraction_methods(self, file_path: str) -> bool:
        """
        loads the calculated extraction methods from a file
        :param file_path: the path to the file to read and parse
        :return: true if the file could be parsed successfully
        """
        if not isfile(file_path):
            return False
        with open(file_path) as file:
            self.EXTRACTION_METHODS = [load(line) for line in file]
            return True

    def save_extraction_methods(self, file_path: str) -> bool:
        """
        saves the extraction methods in a persistent file
        :param file_path: The path to save the file
        :return: true if saving was successful
        """
        with open(file_path, 'w') as file:
            file.writelines([method.save() + '\n' for method in self.EXTRACTION_METHODS])
            return True
