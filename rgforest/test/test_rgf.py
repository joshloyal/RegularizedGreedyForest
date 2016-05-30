import numpy as np

import sklearn.metrics as metrics

from rgforest import rgf
from rgforest import dataset

module_rng = np.random.RandomState(1234)

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


def test_train_classification():
    (X_train, y_train), (X_test, y_test) = get_test_data()
    est = rgf.RegularizedGreedyForestEstimator(l2=0.01, max_leaf_nodes=500, verbose=1)
    est.fit(X_train, y_train)

    y_pred = est.predict(X_train)
    y_pred = (y_pred > 0.5).astype(np.int)
    train_score = metrics.accuracy_score(y_train, y_pred)
    assert train_score > 0.5

    y_pred = est.predict(X_test)
    y_pred = (y_pred > 0.5).astype(np.int)
    test_score = metrics.accuracy_score(y_test, y_pred)
    assert test_score > 0.5


def test_train_regression():
    (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
    est = rgf.RegularizedGreedyForestEstimator(l2=0.01, max_leaf_nodes=500, loss='LS')
    est.fit(X_train, y_train)

    y_pred = est.predict(X_train)
    train_score = metrics.mean_squared_error(y_train, y_pred)
    assert train_score < 2

    y_pred = est.predict(X_test)
    test_score = metrics.mean_squared_error(y_test, y_pred)
    assert test_score < 2

def test_dataset_ones_wide():
    n_features = 100
    n_samples = 10

    X = np.ones((n_samples, n_features)).astype(np.float64)
    y = module_rng.choice([1, -1], n_samples).astype(np.float64)
    rgf_matrix = dataset.RGFMatrix(X, y)
    del X
    del y

    assert rgf_matrix.n_features == n_features
    assert rgf_matrix.n_samples == n_samples

    np.testing.assert_allclose(rgf_matrix.sum_features(), n_features * n_samples)

def test_dataset_ones_long():
    n_features = 10
    n_samples = 100

    X = np.ones((n_samples, n_features)).astype(np.float64)
    y = module_rng.choice([1, -1], n_samples).astype(np.float64)
    rgf_matrix = dataset.RGFMatrix(X, y)
    del X
    del y

    assert rgf_matrix.n_features == n_features
    assert rgf_matrix.n_samples == n_samples

    np.testing.assert_allclose(rgf_matrix.sum_features(), n_features * n_samples)

def test_dataset_ones_equal():
    n_features = 10
    n_samples = 10

    X = np.ones((n_samples, n_features)).astype(np.float64)
    y = module_rng.choice([1, -1], n_samples).astype(np.float64)
    rgf_matrix = dataset.RGFMatrix(X, y)
    del X
    del y

    assert rgf_matrix.n_features == n_features
    assert rgf_matrix.n_samples == n_samples

    np.testing.assert_allclose(rgf_matrix.sum_features(), n_features * n_samples)

def test_dataset_random():
    n_features = 100
    n_samples = 10

    X = module_rng.randn(n_samples, n_features).astype(np.float64)
    y = module_rng.choice([1, -1], n_samples).astype(np.float64)
    rgf_matrix = dataset.RGFMatrix(X, y)

    assert rgf_matrix.n_features == n_features
    assert rgf_matrix.n_samples == n_samples

    np.testing.assert_allclose(rgf_matrix.sum_features(), X.sum())
    for i in range(n_samples):
        np.testing.assert_allclose(rgf_matrix.target_get(i), y[i])
        for j in range(n_features):
            np.testing.assert_allclose(rgf_matrix.feature_get(i, j), X[i, j])
