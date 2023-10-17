import argparse
import json
import numpy as np
import os
import pickle

from tqdm import tqdm

from conformal_predictors import \
    LPRegressionConformalPredictor, ZeroOneClassificationConformalPredictor, \
    CQRRegressionConformalPredictor, LACClassificationConformalPredictor, \
    APSClassificationConformalPredictor, \
    LPHighDimensionRegressionConformalPredictor
from utils import set_random_seed, get_datasets, get_error, get_size

def get_dataset_results(dataset, X, y, args):
    typ = args['type']
    alpha = args['alpha']
    gamma = args['gamma']
    frac_train = args['frac_train']
    frac_cal = args['frac_cal']
    it_train = args['it_train']
    it_cal = args['it_cal']

    # Set random seeds
    set_random_seed()

    # Compute dataset sizes
    n = X.shape[0]
    n_train = round(frac_train * n)
    n_cal = round(frac_cal * n)
    n_test = n - n_train - n_cal

    # Initialize the conformal predictor
    if typ == 'L1Regression':
        predictor = LPRegressionConformalPredictor(y)
    elif typ == 'ZeroOneClassification':
        num_classes = np.unique(y).shape[0]
        predictor = ZeroOneClassificationConformalPredictor(num_classes)
    elif typ == 'CQRRegression':
        predictor = CQRRegressionConformalPredictor(y)
    elif typ == 'LACClassification':
        num_classes = np.unique(y).shape[0]
        predictor = LACClassificationConformalPredictor(num_classes)
    elif typ == 'APSClassification':
        num_classes = np.unique(y).shape[0]
        predictor = APSClassificationConformalPredictor(num_classes)
    elif typ == 'L1HighDimensionRegression':
        predictor = LPHighDimensionRegressionConformalPredictor(y, 1.)
    elif typ == 'L2HighDimensionRegression':
        predictor = LPHighDimensionRegressionConformalPredictor(y, 2.)

    results = {
        'error' : [], 'size' : [], 'estimated size' : [],
        'estimated size (lower)' : [], 'estimated size (upper)' : []
    }

    for _ in tqdm(range(it_train), leave=False):
        idx = np.arange(n)

        # Check the presence of all classes in the classification case
        while True:
            # Randomly sample the training dataset
            np.random.shuffle(idx)
            idx_train = idx[: n_train]
            idx_no_train = idx[n_train :]

            X_train = X[idx_train]
            y_train = y[idx_train]

            if 'Regression' in typ:
                break
            elif 'Classification' in typ:
                if num_classes == np.unique(y_train).shape[0]:
                    break

        # Set the training dataset
        predictor.set_train(X_train, y_train)

        for _ in tqdm(range(it_cal), leave=False):
            # Randomly sample the calibration dataset
            np.random.shuffle(idx_no_train)
            idx_cal = idx_no_train[: n_cal]
            idx_no_train_cal = idx_no_train[n_cal :]

            X_cal = X[idx_cal]
            y_cal = y[idx_cal]

            # Set the calibration dataset
            predictor.set_calibration(X_cal, y_cal, alpha)

            # Compute the estimates for the expected prediction set size
            X_access = X_cal
            y_access = y_cal
            size, size_lower, size_upper = \
                predictor.get_prediction_set_size_estimates(
                    X_access, y_access, gamma
                )
            results['estimated size'].append(size)
            results['estimated size (lower)'].append(size_lower)
            results['estimated size (upper)'].append(size_upper)

            # Sample the remaining test dataset
            idx_test = idx_no_train_cal

            X_test = X[idx_test]
            y_test = y[idx_test]

            # Compute the prediction set and its corresponding error and size
            C = predictor.get_prediction(X_test)
            error = get_error(typ, C, y_test)
            size = get_size(typ, C)
            results['error'].append(error.tolist())
            results['size'].append(size.tolist())

        # Accumulate the results
        stat = {
            'type'              : typ,
            'dataset'           : dataset,
            'alpha'             : alpha,
            'gamma'             : gamma,
            'num_datapoints'    : n,
            'num_features'      : X.shape[1],
            'num_train'         : n_train,
            'num_cal'           : n_cal,
            'num_test'          : n_test,
            'it_train'          : it_train,
            'it_cal'            : it_cal,
            'results'           : results
        }
        if 'Regression' in typ:
            stat['y_range'] = [y.min(), y.max()]
        elif 'Classification' in typ:
            stat['num_classes'] = num_classes

        # Save the results
        os.makedirs('results', exist_ok=True)
        name = {'type' : typ, 'dataset' : dataset}
        file_path = os.path.join('./results', '%s.pkl' % json.dumps(name))
        file = open(file_path, 'wb')
        pickle.dump(stat, file)
        file.close()

def get_results(args):
    # Run each dataset individually
    for dataset, X, y in tqdm(get_datasets(args['type']), leave=False):
        get_dataset_results(dataset, X, y, args)

def main():
    # Get inputs
    parser = argparse.ArgumentParser(
        description='On the Expected Size of Conformal Prediction Sets'
    )
    parser.add_argument(
        '--type', type=str, required=True,
        choices=[
            'L1Regression', 'ZeroOneClassification', 'CQRRegression',
            'LACClassification', 'APSClassification',
            'L1HighDimensionRegression', 'L2HighDimensionRegression'
        ],
        help='Conformal predictor type'
    )
    parser.add_argument(
        '--alpha', type=float, default=0.1,
        help='Conformal predictor significance level'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.1,
        help='Prediction set size interval significance level'
    )
    parser.add_argument(
        '--frac_train', type=float, default=0.25,
        help='Fraction used as training dataset'
    )
    parser.add_argument(
        '--frac_cal', type=float, default=0.25,
        help='Fraction used as calibration dataset'
    )
    parser.add_argument(
        '--it_train', type=int, default=10,
        help='Number of iterations for sampling a new training dataset'
    )
    parser.add_argument(
        '--it_cal', type=int, default=100,
        help='Number of iterations for sampling a new calibration dataset'
    )
    args = vars(parser.parse_args())

    # Run the main code
    get_results(args)

if __name__ == '__main__':
    main()

