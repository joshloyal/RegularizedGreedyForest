import os
import cPickle as pickle

import tempfile
import numpy as np
import sklearn.metrics as metrics

import rgforest as rgf
from rgforest.tests.test_utils import get_test_data, get_fixture_path


module_rng = np.random.RandomState(1234)


class TestRegularizedGreedyForest(object):
    """Test suite for regularized greedy forest """
    def test_train_classification(self):
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


    def test_train_classification_weighted(self):
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

    def test_classification_regression(self):
        (X_train, y_train), (X_test, y_test) = get_test_data()
        sample_weight = np.ones_like(y_train)
        sample_weight[y_train == 0] = 0.5

        est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train, sample_weight)

        y_pred = est.predict(X_train)

        fixture_name = get_fixture_path('rgf_classification_regression.npy')
        saved_preds = np.loadtxt(fixture_name)
        np.testing.assert_allclose(y_pred, saved_preds)

    def test_train_regression(self):
        (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
        est = rgf.RegularizedGreedyForestRegressor(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        y_pred = est.predict(X_train)
        train_score = metrics.mean_squared_error(y_train, y_pred)
        assert train_score < 2

        y_pred = est.predict(X_test)
        test_score = metrics.mean_squared_error(y_test, y_pred)
        assert test_score < 2

    def test_train_regression_weighted(self):
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

    def test_train_regression_regression(self):
        (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
        est = rgf.RegularizedGreedyForestRegressor(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        y_pred = est.predict(X_train)

        fixture_name = get_fixture_path('rgf_regression_regression.npy')
        saved_preds = np.loadtxt(fixture_name)
        np.testing.assert_allclose(y_pred, saved_preds)

    def test_pickle_classifier(self):
        (X_train, y_train), (X_test, y_test) = get_test_data()
        est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        y_pred = est.predict_proba(X_test)
        with tempfile.NamedTemporaryFile('wr') as tfile:
            file_name = tfile.name

        try:
            pickle.dump(est, open(file_name, 'wb'))
            unpickle_est = pickle.load(open(file_name, 'rb'))
            unpickle_y_pred = unpickle_est.predict_proba(X_test)
            np.testing.assert_allclose(y_pred, unpickle_y_pred)
        finally:
            os.remove(file_name)

    def test_pickle_regressor(self):
        (X_train, y_train), (X_test, y_test) = get_test_data(classification=False)
        est = rgf.RegularizedGreedyForestRegressor(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        y_pred = est.predict(X_test)

        with tempfile.NamedTemporaryFile('wr') as tfile:
            file_name = tfile.name

        try:
            pickle.dump(est, open(file_name, 'wb'))
            unpickle_est = pickle.load(open(file_name, 'rb'))
            unpickle_y_pred = unpickle_est.predict(X_test)
            np.testing.assert_allclose(y_pred, unpickle_y_pred)
        finally:
            os.remove(file_name)
