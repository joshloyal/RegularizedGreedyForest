import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from ._ensemble import RGFBuilder, RGFTreeEnsemble


def check_weight(sample_weight):
    if np.any(sample_weight == 0):
        raise ValueError('`sample_weight` must be non-zero')


class RegularizedGreedyForestEstimator(BaseEstimator):
    def __init__(self, max_leaf_nodes=500, l2=1,  loss='LS',
                 test_interval=100, verbose=False):
        self.max_leaf_nodes = max_leaf_nodes
        self.l2 = l2
        self.loss = loss
        self.test_interval = test_interval
        self.verbose = verbose

        self.ensemble = RGFTreeEnsemble()

    def _fit(self, X, y, sample_weight=None):
        check_weight(sample_weight)
        builder = RGFBuilder(self.max_leaf_nodes, self.l2, self.loss,
                             self.test_interval, self.verbose)
        builder.build(self.ensemble, X, y, sample_weight)

    def fit(self, X, y, sample_weight=None):
        self._fit(X, y, sample_weight)

    def predict(self, X):
        return self.ensemble.predict(X)

    def save(self, file_name):
        self.ensemble.save(file_name)


class RegularizedGreedyForestClassifier(RegularizedGreedyForestEstimator,
                                        ClassifierMixin):
    def __init__(self, max_leaf_nodes=500, l2=1,  loss='Log',
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
        return self.ensemble.predict(X)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return self.decision_function(y_proba)


class RegularizedGreedyForestRegressor(RegularizedGreedyForestEstimator,
                                       RegressorMixin):

    def __init__(self, max_leaf_nodes=500, l2=1,
                 test_interval=100, verbose=0):
        super(RegularizedGreedyForestRegressor, self).__init__(
            max_leaf_nodes=max_leaf_nodes,
            l2=l2,
            loss='LS',  # hard-code least squares loss
            test_interval=test_interval,
            verbose=verbose)
