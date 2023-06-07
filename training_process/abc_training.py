import datetime
import time

import numpy as np

from classification.abc_classifier import ABCClassifier
from training_process.iftsvm_training import calculate_measures


class ABCTrainer:

    def __init__(self, dataset: [], positive_labels: [int]):
        self.dataset = dataset
        self.positive_labels = positive_labels

    def find_best_coefficients(self, path=None):
        """
        Tries to find the best coefficients from a predefined list of hyperparameter values
        :return: Trained ABC Classifier which had the best score
        """
        start = time.time()
        print('start training')
        np.random.shuffle(self.dataset)
        methods = ['linear', 'non-linear']
        generations = [100, 200]
        population_size = [1000, 2000]
        cycle_numbers = [500, 1000]
        # currently not in use
        fit_neighbors = [5]
        chosen_number = [5, 10, 20]
        kernel_size = [5, 7, 20]
        sight = [5, 10]
        fit_functions = ['count', 'sight']
        train_data = np.array_split(self.dataset, 100)
        best = None
        best_score = 0
        for gen in generations:
            for pop in population_size:
                for cyc in cycle_numbers:
                    for fit in fit_neighbors:
                        for chosen in chosen_number:
                            for s in sight:
                                for method in methods:
                                    for ksi in kernel_size:
                                        for fit_func in fit_functions:
                                            if method == 'linear' and ksi is not kernel_size[0]:
                                                continue
                                            classifier = ABCClassifier(method=method, generations=gen,
                                                                       population_size=pop, cycle_numbers=cyc,
                                                                       fit_neighbors=fit, chosen_number=chosen,
                                                                       kernel_size=ksi, sight=s, fit_function=fit_func)
                                            classifier.train(train_data[0], self.positive_labels)
                                            far, acc, dr = calculate_measures(classifier, np.concatenate(train_data[1:20]))
                                            print(
                                                f'generations={gen}, population_size={pop}, cycle_numbers={cyc}, fit_neighbors={fit}, chosen_number={chosen}, kernel_size={ksi}, sight={s}, fit_func={fit_func}')
                                            print(f'radii({len(classifier.radii)})={classifier.radii}')
                                            print(f'FAR: {far}, Accuracy: {acc}, Detection Rate: {dr}')
                                            if acc + dr - far > best_score:
                                                best = classifier
                                                best_score = acc + dr - far
        print('best result:')
        print(
            f'method={best.method}, generations={best.generations}, population_size={best.population_size}, cycle_numbers={best.cycle_numbers}, fit_neighbors={best.fit_neighbors}, chosen_number={best.chosen_number}, kernel_size={best.kernel_size}, sight={best.sight}, fit_func={best.fit_function}')
        print(f'radii({len(best.radii)})={best.radii}')
        far, acc, dr = calculate_measures(best, np.concatenate(train_data[1:20]))
        print(f'FAR: {far}, Accuracy: {acc}, Detection Rate: {dr}')
        if path:
            best.save(path)
        print(f' time used: {datetime.timedelta(seconds=int(time.time() - start))}')
        return best
