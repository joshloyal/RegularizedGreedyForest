import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from rgforest._ensemble import RGFTreeEnsemble
from rgforest._builder import RGFBuilder
from rgforest.exceptions import NotFittedError


def expit(x):
    return 1. / (1. + np.exp(-x))


def check_weight(sample_weight):
    if np.any(sample_weight == 0):
        raise ValueError('`sample_weight` must be non-zero')


class RegularizedGreedyForestEstimator(BaseEstimator):
    def __init__(self, max_leaf_nodes=500, l2=0.01,  loss='LS',
                 test_interval=100, verbose=False):
        self.max_leaf_nodes = max_leaf_nodes
        self.l2 = l2
        self.loss = loss
        self.test_interval = test_interval
        self.verbose = verbose

        self.ensemble = None

    def _fit(self, X, y, sample_weight=None):
        check_weight(sample_weight)
        # need to check X is c-contigous!

        if self.ensemble is None:
            self.ensemble = RGFTreeEnsemble()

        builder = RGFBuilder(self.max_leaf_nodes, self.l2, self.loss,
                             self.test_interval, self.verbose)
        builder.build(self.ensemble, X, y, sample_weight)

    def fit(self, X, y, sample_weight=None):
        self._fit(X, y, sample_weight)

    def _validate_X_predict(self, X):
        if self.ensemble is None or len(self.ensemble) == 0:
            raise NotFittedError("Estimator not fitted, "
                                 "all `fit` before exploiting the model.")
        return X

    def predict(self, X):
        X = self._validate_X_predict(X)
        return self.ensemble.predict(X)


class RegularizedGreedyForestClassifier(RegularizedGreedyForestEstimator,
                                        ClassifierMixin):
    def __init__(self, max_leaf_nodes=500, l2=0.01,  loss='Log',
                 test_interval=100, verbose=0):
        super(RegularizedGreedyForestClassifier, self).__init__(
            max_leaf_nodes=max_leaf_nodes,
            l2=l2,
            loss=loss,
            test_interval=test_interval,
            verbose=verbose)

    def _transform_target(self, y):
        """The transform targets from {0, 1} to {-1, 1}"""
        return (y * 2 - 1).astype(np.float64)

    def fit(self, X, y, sample_weight=None):
        y = self._transform_target(y)
        self._fit(X, y, sample_weight)

    def decision_function(self, y):
        return (y > 0.5).astype(np.int32)

    def predict_proba(self, X):
        """ we want positive probabilities. unsure what exactly rgf does """
        X = self._validate_X_predict(X)
        return expit(2 * self.ensemble.predict(X))

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return self.decision_function(y_proba)


class RegularizedGreedyForestRegressor(RegularizedGreedyForestEstimator,
                                       RegressorMixin):

    def __init__(self, max_leaf_nodes=500, l2=0.01,
                 test_interval=100, verbose=0):
        super(RegularizedGreedyForestRegressor, self).__init__(
            max_leaf_nodes=max_leaf_nodes,
            l2=l2,
            loss='LS',  # hard-code least squares loss
            test_interval=test_interval,
            verbose=verbose)
