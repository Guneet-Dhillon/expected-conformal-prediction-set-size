import math
import numpy as np
import random

from ucimlr import classification_datasets, regression_datasets

def set_random_seed(seed=42):
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)

def get_regression_datasets():
    # Get UCI regression datasets
    datasets = []
    for dataset in regression_datasets.all_datasets():
        train = dataset('data', split='train', validation_size=0.)
        test = dataset('data', split='test', validation_size=0.)

        name = train.name
        X = np.vstack([train.x, test.x])
        y = np.vstack([train.y, test.y])

        if X.shape[0] >= 1000 and y.shape[1] == 1:
            y = y.squeeze()

            dataset = (name, X, y)
            datasets.append(dataset)

    return datasets

def get_high_dimension_regression_datasets():
    # Get UCI high-dimension regression datasets
    datasets = []
    for dataset in regression_datasets.all_datasets():
        train = dataset('data', split='train', validation_size=0.)
        test = dataset('data', split='test', validation_size=0.)

        name = train.name
        X = np.vstack([train.x, test.x])
        y = np.vstack([train.y, test.y])

        if X.shape[0] >= 1000 and y.shape[1] > 1:
            dataset = (name, X, y)
            datasets.append(dataset)

    return datasets

def get_classification_datasets():
    # Get UCI classification datasets
    datasets = []
    for dataset in classification_datasets.all_datasets():
        train = dataset('data', split='train', validation_size=0.)
        test = dataset('data', split='test', validation_size=0.)

        name = train.name
        X = np.vstack([train.x, test.x])
        y = np.hstack([train.y, test.y])

        if X.shape[0] >= 1000:
            dataset = (name, X, y)
            datasets.append(dataset)

    return datasets

def get_datasets(typ):
    if 'Regression' in typ:
        if 'HighDimension' in typ:
            return get_high_dimension_regression_datasets()
        return get_regression_datasets()
    elif 'Classification'in typ:
        return get_classification_datasets()

def get_regression_error(C, y):
    # Compute prediction error in the regression case
    def isclose(xs, ys):
        close = np.array([math.isclose(x, y) for x, y in zip(xs, ys)])
        return close

    close = np.logical_or(isclose(C[0], y), isclose(C[1], y))
    contain = np.logical_and(C[0] <= y, y <= C[1])

    error = 1. - np.logical_or(close, contain)
    return error

def get_high_dimension_regression_error(C, y):
    # Compute prediction error in the high-dimension regression case
    yh, r, p = C
    error = 1. - ((np.abs(yh - y) ** p).sum(1) <= r)
    return error

def get_classification_error(C, y):
    # Compute prediction error in the classification case
    n = C.shape[0]
    error = (1. - C[np.arange(n), y])
    return error

def get_error(typ, C, y):
    if 'Regression' in typ:
        if 'HighDimension' in typ:
            return get_high_dimension_regression_error(C, y)
        return get_regression_error(C, y)
    elif 'Classification'in typ:
        return get_classification_error(C, y)

def get_regression_size(C):
    # Compute prediction set size in the regression case
    size = (C[1] - C[0])
    return size

def get_high_dimension_regression_size(C):
    # Compute prediction set size in the high-dimension regression case
    yh, r, p = C
    d = yh.shape[1]
    size = \
        (r ** (d / p)) * ((2. * math.gamma((1. / p) + 1.)) ** d) \
        / math.gamma((d / p) + 1.)
    return size

def get_classification_size(C):
    # Compute prediction set size in the classification case
    size = C.sum(1)
    return size

def get_size(typ, C):
    if 'Regression' in typ:
        if 'HighDimension' in typ:
            return get_high_dimension_regression_size(C)
        return get_regression_size(C)
    elif 'Classification'in typ:
        return get_classification_size(C)

def get_size_error(size, size_lower, size_upper):
    # Compute prediction set size interval error
    def isclose(xs, y):
        close = np.array([math.isclose(x, y) for x in xs])
        return close

    close = np.logical_or(isclose(size_lower, size), isclose(size_upper, size))
    contain = np.logical_and(size_lower <= size, size <= size_upper)

    size_error = 1. - np.logical_or(close, contain)
    return size_error

def get_mean_std(x):
    # Compute the mean and the standard deviation
    mean = x.mean()
    std = x.std(ddof=1)
    return mean, std

