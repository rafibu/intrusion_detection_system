import statistics
from ast import literal_eval
from math import log


class ExtractionMethod:
    """
    This class works as a wrapper for extraction methods
    """

    def transform(self, sample: str):
        """
        This method transforms a value according to the rules of the extraction method
        :param sample: the value to transform
        :return: the transformed value
        """
        pass

    def save(self):
        """
        This method should return a string which describes the object completely
        """
        pass


class ScaleToRange(ExtractionMethod):
    x_min: float
    x_max: float

    def __init__(self, values):
        self.x_min = min(values)
        self.x_max = max(values)

    def transform(self, sample):
        sample = float(sample)
        if sample < self.x_min:
            self.x_min = sample
        if sample > self.x_max:
            self.x_max = sample
        return (sample - self.x_min) / (self.x_max - self.x_min)

    def __str__(self):
        return f'STR min: {self.x_min}, max: {self.x_max}'

    def save(self):
        return f'STR:{self.x_min},{self.x_max}'


class LogScaling(ExtractionMethod):

    def transform(self, sample):
        sample = float(sample)
        if sample <= 0:
            return 0
        return log(sample)

    def __str__(self):
        return 'log'

    def save(self):
        return 'LOG'


class ZScale(ExtractionMethod):
    mean: float
    deviation: float

    def __init__(self, *args, **kwargs):
        if 'mean' in kwargs:
            # loaded from a saved file
            self.mean = kwargs['mean']
            self.deviation = kwargs['dev']
        else:
            values = args[0]
            self.mean = sum(values) / len(values)
            self.deviation = statistics.stdev(values)

    def transform(self, sample):
        sample = float(sample)
        return (sample - self.mean) / self.deviation

    def __str__(self):
        return f'z-sc mean: {self.mean} dev: {self.deviation}'

    def save(self):
        return f'ZSC:{self.mean},{self.deviation}'


class OneHotEncoding(ExtractionMethod):
    dictionary: dict
    counter: int

    def __init__(self, **kwargs):
        if kwargs.get('data') is not None:
            # initialize counter and add all data pairs to the dictionary
            self.counter = 0
            self.dictionary = {}
            [self.transform(data) for data in kwargs.get('data')]
        elif kwargs.get('dictionary') is not None:
            # loaded from saved file
            self.dictionary = kwargs.get('dictionary')
            self.counter = max(self.dictionary.values())
        else:
            raise Exception(f'No valid initialization method in args:{kwargs}')

    def get_and_update_counter(self):
        self.counter += 1
        return self.counter

    def transform(self, sample):
        if sample in self.dictionary:
            return self.dictionary.get(sample)
        count = self.get_and_update_counter()
        self.dictionary[sample] = count
        return count

    def __str__(self):
        return f'ohe dic-len: {len(self.dictionary)}'

    def save(self):
        return f'OHE:{self.dictionary}'


class BooleanEncoding(ExtractionMethod):

    def transform(self, sample):
        return int(sample)

    def __str__(self):
        return 'bool'

    def save(self):
        return 'BOL'


class IPv4Encoding(ExtractionMethod):
    """
    The first two and the second two parts of the IP are encoded separately, as the first two describe the network
    and the second two the host (see https://docs.oracle.com/cd/E19455-01/806-0916/6ja85399u/index.html)
    """
    dictionaries: []
    counter: []

    def __init__(self, *values, **kwargs):
        if len(kwargs) > 0:
            # loaded from saved file
            self.dictionaries = kwargs.get('dictionaries')
            self.counter = [max(self.dictionaries[0].values()), max(self.dictionaries[1].values())]
        else:
            self.counter = [0, 0]
            split_values = [value.split(".") for value in values]
            for i in range(2):
                self.dictionaries[i] = {f'{value[2 * i]}.{value[2 * i + 1]}': self.get_and_update_counter(i) for value
                                        in split_values}

    def get_and_update_counter(self, i: int):
        self.counter[i] += 1
        return self.counter[i]

    def transform(self, sample):
        parts = sample.split('.')
        network = self.find_or_update(f'{parts[0]}.{parts[1]}', 0)
        host = self.find_or_update(f'{parts[2]}.{parts[3]}', 1)
        return network, host

    def __str__(self):
        return f'ipv4 loc-len: {len(self.dictionaries[0])} host-len: {len(self.dictionaries[1])}'

    def save(self):
        return f'IPV4:{self.dictionaries[0]};{self.dictionaries[1]}'

    def find_or_update(self, key: str, i: int):
        if key not in self.dictionaries[i]:
            self.dictionaries[i][key] = self.get_and_update_counter(i)
        return self.dictionaries[i][key]


def load(description: str) -> ExtractionMethod:
    """
    This method should return the object described by the given line
    :param description: string which describes the object
    :return: ExtractionMethod object
    """
    des = description.split(':')
    match des[0].strip():
        case 'STR':
            args = des[1].split(',')
            return ScaleToRange([float(num) for num in args])
        case 'LOG':
            return LogScaling()
        case 'ZSC':
            args = des[1].split(',')
            return ZScale(mean=float(args[0]), dev=float(args[1]))
        case 'OHE':
            return OneHotEncoding(dictionary=literal_eval(description[4:]))
        case 'IPV4':
            dictionaries = description[5:].split(';')
            assert len(dictionaries) == 2
            return IPv4Encoding(dictionaries=dictionaries)
        case 'BOL':
            return BooleanEncoding()
