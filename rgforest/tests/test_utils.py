import os
import numpy as np


TEST_DIR = os.path.abspath(os.path.dirname(__file__))


def get_test_data(n_train=1000, n_test=500, n_features=10, n_categoricals=0,
                  n_classes=2, classification=True, random_seed=1234):
    """Classification or regression problem that should be easily solvable by a gbm

    Parameters
    ----------
    n_train : int
        number of training examples
    n_test : int
        number of test examples
    n_features : int
        number of features in the synthetic dataset
    n_categorical : int
        number of categorical features to be present
    n_classes : int
        number of classes for a classification target
    classification : bool (default=True)
        Classification problem if True, otherwise regression
    random_seed : int
        Seed for the random number generator
    """
    rng = np.random.RandomState(random_seed)
    n_samples = n_train + n_test

    n_features = n_features - n_categoricals
    if n_features < 0:
        raise ValueError('Requested more categoricals than n_features')

    if classification:
        #  guassian blobs
        y = rng.randint(0, n_classes, size=(n_samples,)).astype(np.float64)
        X = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            X[i] = rng.normal(loc=y[i], scale=0.7, size=(n_features,))
    else:
        y_loc = rng.random_sample((n_samples,))
        X = np.zeros((n_samples, n_features))
        y = np.zeros((n_samples, 1)).astype(np.float64)
        for i in range(n_samples):
            X[i] = rng.normal(loc=y_loc[i], scale=0.7, size=(n_features,))
            y[i] = rng.normal(loc=y_loc[i], scale=0.7, size=(1,))
        y = y.ravel()

    if n_categoricals:
        for _ in range(n_categoricals):
            X = np.hstack((X, rng.choice(range(10), n_samples)[:, np.newaxis]))

    return (X[:n_train], y[:n_train]), (X[n_train:], y[n_train:])


def get_fixture_path(fixture_name):
    return os.path.join(TEST_DIR, 'fixtures', fixture_name)
