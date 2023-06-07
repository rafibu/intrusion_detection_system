from classification.classifier import Classifier
from feature_selection.genetic_algorithm_selector import reduce_features


class TrainedModel:
    """
    This class represents a finished trained model with different Classifiers and a Decider. The classifiers are either
    one or two of the same kind. If only two have been trained, they act as normal/attack classifiers. If only one is
    trained, it'll be used as the attack classifier.
    """

    def __init__(self, classifiers: [Classifier], features: [int], decider=None):
        """
        :param classifiers: the classifiers, ordered by their effectiveness, if two classifiers are held of the same
        kind, they should have the order [normal_classifier, attack_classifier]
        :param features: the features which should be used during the classification
        :param decider: Decider which should be used, if none is chosen, the default Decider is used
        """
        self.classifiers = classifiers
        if decider is not None:
            self.decider = decider
        else:
            self.decider = Decider()
        self.features = features

    def classify(self, sample: []) -> tuple:
        """
        :param sample: the sample which should be classified
        :return: a tuple, where the first entry is 0 for attack or 1 for normal and the second entry is the membership
                degree of this sample
        """
        reduced_sample = reduce_features(sample, self.features)
        all_mem_degs = []
        for classifier in self.classifiers:
            mem_deg = [c.classify(reduced_sample) for c in classifier]
            sufficient, label = self.decider.decision(mem_deg)
            if sufficient:
                if len(mem_deg) == 2:
                    return label, max(mem_deg)
                else:
                    return label, abs(label - max(mem_deg))
            all_mem_degs.append(mem_deg)
        # If no classifier could classify sufficiently enough
        attack = []
        normal = []
        for mem_deg in all_mem_degs:
            if len(mem_deg) == 2:
                normal.append(mem_deg[0])
                attack.append(mem_deg[1])
            else:
                attack.append(mem_deg[0])
                normal.append(1 - mem_deg[0])
        att = sum(attack) / len(attack)
        nor = sum(normal) / len(normal)
        return int(nor > att), max(att, nor)


class Decider:
    """
    The decider takes one or two membership degrees into account. If two are given the following two inequalities have
    to be true to have a decision, otherwise the next classifier has to be used.

    * |µ_att(x) - µ_nor(x)| > t_diff
    * max(µ_att(x), µ_nor(x)) > t_max

    if only one membership degree is present, only the second inequality is checked
    """

    def __init__(self, t_diff=0.1, t_max=0.7):
        """
        :param t_diff: threshold for the difference between the two membership degrees
        :param t_max: threshold for the membership degree to surpass
        """
        self.t_diff = t_diff
        self.t_max = t_max

    def decision(self, classifications: []) -> (bool, int):
        """
        Decides if the classifications are sufficient to create a clear result
        :param classifications: membership degrees of the classifications
        :return: true if the classification is clear enough and 1 if it is more likely an attack, 0 otherwise
        """
        if len(classifications) > 2:
            raise RuntimeError(
                f'Cannot have more than two classifications, currently {len(classifications)} classifications')
        if len(classifications) == 2:
            if abs(classifications[0] - classifications[1]) <= self.t_diff:
                return False, classifications.index(max(classifications))
            return max(classifications) > self.t_max, classifications.index(max(classifications))
        if max(classifications) > self.t_max:
            return True, 0
        elif 1 - max(classifications) > self.t_max:
            return True, 1
        return False, int(max(classifications) < 1 - max(classifications))
