import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from _rgf import RegularizedGreedyForest


double_t = np.float64

PARAM_FMT = ('reg_L2={l2},loss={loss},test_interval={test_interval},'
             'max_leaf_forest={max_leaf_forest}')

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

        self._trainer = RegularizedGreedyForest(
            self.param_str, int(self.verbose))

    @property
    def param_str(self):
        parameters = PARAM_FMT.format(
            l2=self.l2,
            loss=self.loss,
            test_interval=self.test_interval,
            max_leaf_forest=self.max_leaf_nodes)
        if self.verbose:
            parameters += ',Verbose'
        return parameters

    def _fit(self, X, y, sample_weight=None):
        check_weight(sample_weight)
        self._trainer.fit(X, y, sample_weight)

    def fit(self, X, y, sample_weight=None):
        self._fit(X, y, sample_weight)

    def predict(self, X):
        return self._trainer.predict(X)

    def save(self, file_name):
        self._trainer.save_model(file_name)


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
        return self._trainer.predict(X)

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
