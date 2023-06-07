# Fuzzy Intrusion Detection System

---
This project contains the source code to Rafael Burkhalters master thesis, written at the University of Fribourg CH in 2023.
The IDS can be trained on data sets and then predict if a new network traffic package is malicious or not.
## Feature Extraction

---
The feature extraction process takes a list of samples and normalize them into numbers.
To use the feature extraction, first it has to be trained on a representable training set. The methods can be saved to a CSV file.

    feature_extractor = FeatureExtractor()
    feature_extractor.calculate_extraction_methods(data)
    feature_extractor.save_extraction_methods(feature_methods_path)

Afterward, samples can be transformed with the trained feature extractor
 
    transformed_sample = feature_extractor.transform(sample)

## Feature Selection

---
Feature selection processes try to find the most important features which identify the features with the highest correlation to attacks.
In this project two feature selectors are implemented, the CorrelationSelector, based on pandas correlation method, and the GenAlgSelector, based on the genetic algorithm.
They can both be trained and used in an analogous way. They need to know which labels are considered an attack to adequately find a correlation:
    
    correlation_selector = CorrelationSelector()
    correlation_selector.train(training_data, positive_labels)

There are two possibilities on how to get a list of features from a trained feature selector, one gives the best features as a list of fixed length, the other gives all features which have a correlation over a certain threshold. The threshold should be in [0,1]:

    features = correlation_selector.get_with_threshold(threshold)
    features = correlation_selector.get_highest_ranked_features(num_of_features)

The features can then be used to either reduce them for a list of data

    corr_data = reduce_bulk_features(training_data, features)

or just one sample
    
    corr_sample = reduce_features(sample, features)

## Classification

---
The classification works similarly as the feature selection. Data should already be normalized and reduced to the most important features.
There are currently two classifiers implemented. The methods are called in the same way, the classification algorithm differs however.
The first algorithm uses an IFTSVM, the other a fuzzyfied ABC algorithm.

    classifier = IFTSVMClassifier()
    classifier.train(corr_data, positive_labels)

The trained classifier can also be saved and loaded to and from a CSV file

    classifier.save(path)
    classifier.load(path)

The classifier calculates the membership degree of a given sample, the result lies between 0 and 1:
    
    result = classifier.classify(sample)

## Hyperparameter Optimization

To simplify the process of finiding the best hyperparameters for each classifier, two Training classes can be used. They try to find an optimal permutation of hyperparameters from a finite set of possibilities.

    iftsvm_trainer = IFTSVMTrainer(corr_data, positive_labels, 'non-linear')
    iftsvm_classifier = iftsvm_trainer.find_best_coefficients()

The resulting classifier is only trained on a fraction of the data and should be trained on the whole training set again.

## Trained Model

To create a complete IDS which uses several classifiers and feature selectors, a TrainedModel object can be created. 
This model will be trained on an optimized fraction of the training data and create a ranked list of classifiers which will be used in the classification process to create a clear result.

    classifiers = [
        IFTSVMClassifier(),
        ABCClassifier()
    ]
    trained_model = train_model(classifiers=classifiers, data=data, positive_labels=positive_labels)