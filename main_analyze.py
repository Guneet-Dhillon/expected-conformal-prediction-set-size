import numpy as np
import os
import pickle

from tqdm import tqdm

from utils import get_size_error, get_mean_std

def get_stats():
    stats = {}
    for file_name in tqdm(sorted(os.listdir('./results')), leave=False):
        # Get the saved results
        file_path = os.path.join('./results', file_name)
        file = open(file_path, 'rb')
        stat_in = pickle.load(file)
        file.close()

        # Get the saved stats
        typ = stat_in['type']
        error = np.array(stat_in['results']['error'])
        size = np.array(stat_in['results']['size'])
        size_est = np.array(stat_in['results']['estimated size'])
        size_est_lower = np.array(stat_in['results']['estimated size (lower)'])
        size_est_upper = np.array(stat_in['results']['estimated size (upper)'])

        # Compute the mean and the standard deviation of the stats
        error = get_mean_std(error)
        size = get_mean_std(size)

        # Compute the prediction set size interval error
        error_size_est = get_size_error(size[0], size_est_lower, size_est_upper)

        # Compute the mean and the standard deviation of the stats
        size_est = get_mean_std(size_est)
        size_est_lower = get_mean_std(size_est_lower)
        size_est_upper = get_mean_std(size_est_upper)
        error_size_est = get_mean_std(error_size_est)

        # Accumulate the results
        stat_out = {
            'dataset'                           : stat_in['dataset'],
            'num_datapoints'                    : stat_in['num_datapoints'],
            'num_features'                      : stat_in['num_features'],
            'error'                             : error,
            'size'                              : size,
            'estimated size'                    : size_est,
            'estimated size (lower)'            : size_est_lower,
            'estimated size (upper)'            : size_est_upper,
            'error (estimated size interval)'   : error_size_est
        }
        if 'Regression' in typ:
            stat_out['y_range'] = stat_in['y_range']
        elif 'Classification' in typ:
            stat_out['num_classes'] = stat_in['num_classes']

        # Sort the results by the conformal predictor type
        if typ not in stats:
            stats[typ] = []
        stats[typ].append(stat_out)

    return stats

def print_stats(stats):
    for typ in stats.keys():
        print('Type : %s' % typ)
        print()

        # Print the dataset stats
        print('Dataset')
        if 'Regression' in typ:
            print(
                'Dataset \t Number of data points \t Number of features '
                '\t Label range'
            )
            for stat in stats[typ]:
                print('%s \t %d \t %d \t [%.4f, %.4f]' % (
                    stat['dataset'], stat['num_datapoints'],
                    stat['num_features'], *stat['y_range']
                ))
        elif 'Classification' in typ:
            print(
                'Dataset \t Number of data points \t Number of features '
                '\t Number of classes'
            )
            for stat in stats[typ]:
                print('%s \t %d \t %d \t %d' % (
                    stat['dataset'], stat['num_datapoints'],
                    stat['num_features'], stat['num_classes']
                ))
        print()

        # Print the conformal prediction coverage stats
        print('Coverage')
        print('Dataset \t Error')
        for stat in stats[typ]:
            print('%s \t %.4f' % (stat['dataset'], stat['error'][0]))
        print()

        # Print the conformal prediction set size stats
        print('Prediction set size')
        print(
            'Dataset \t Interval lower bound \t Observed \t Estimated '
            '\t Interval upper bound \t Interval error'
        )
        for stat in stats[typ]:
            print(
                '%s \t %.2f (%.2f) \t %.2f (%.2f) \t %.2f (%.2f) '
                '\t %.2f (%.2f) \t %.4f' % (
                    stat['dataset'], *stat['estimated size (lower)'],
                    *stat['size'], *stat['estimated size'],
                    *stat['estimated size (upper)'],
                    stat['error (estimated size interval)'][0]
            ))
        print()

def main():
    stats = get_stats()
    print_stats(stats)

if __name__ == '__main__':
    main()

