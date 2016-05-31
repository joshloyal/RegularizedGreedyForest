import numpy as np
import os

import sklearn.metrics as metrics

import rgforest as rgf
from rgforest import dataset
from rgforest.tests.test_utils import get_test_data

module_rng = np.random.RandomState(1234)



def test_train_classification():
    (X_train, y_train), (X_test, y_test) = get_test_data()
    est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
    est.fit(X_train, y_train)

    y_pred = est.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred)
    assert train_score > 0.75

    y_pred = est.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred)
    assert test_score > 0.75

    y_proba = est.predict_proba(X_test)
    test_score = metrics.roc_auc_score(y_test, y_proba)
    assert test_score > 0.75


def test_train_classification_weighted():
    (X_train, y_train), (X_test, y_test) = get_test_data()
    sample_weight = np.ones_like(y_train)
    sample_weight[y_train == 0] = 0.5

    est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
    est.fit(X_train, y_train, sample_weight)

    y_pred = est.predict(X_train)
    train_score = metrics.accuracy_score(y_train, y_pred)
    assert train_score > 0.75

    y_pred = est.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred)
    assert test_score > 0.75

    y_proba = est.predict_proba(X_test)
    test_score = metrics.roc_auc_score(y_test, y_proba)
    assert test_score > 0.75


def test_train_regression():
    (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
    est = rgf.RegularizedGreedyForestRegressor(l2=0.01, max_leaf_nodes=500)
    est.fit(X_train, y_train)

    y_pred = est.predict(X_train)
    train_score = metrics.mean_squared_error(y_train, y_pred)
    assert train_score < 2

    y_pred = est.predict(X_test)
    test_score = metrics.mean_squared_error(y_test, y_pred)
    assert test_score < 2

def test_train_regression_weighted():
    (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
    sample_weight = np.ones_like(y_train)
    sample_weight[y_train < 0] = 0.5

    est = rgf.RegularizedGreedyForestRegressor(l2=0.01, max_leaf_nodes=500)
    est.fit(X_train, y_train, sample_weight)

    y_pred = est.predict(X_train)
    train_score = metrics.mean_squared_error(y_train, y_pred)
    assert train_score < 2

    y_pred = est.predict(X_test)
    test_score = metrics.mean_squared_error(y_test, y_pred)
    assert test_score < 2


def test_save_model():
    (X_train, y_train), (X_test, y_test) = get_test_data()
    est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
    est.fit(X_train, y_train)

    file_name = 'test_model'
    try:
        est.save(file_name)
    finally:
        os.remove(file_name)


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
