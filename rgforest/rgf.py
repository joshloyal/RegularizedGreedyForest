from sklearn.base import BaseEstimator

from _rgf import RegularizedGreedyForest


PARAM_FMT = ('reg_L2={l2},loss={loss},test_interval={test_interval},'
             'max_leaf_forest={max_leaf_forest}')


class RegularizedGreedyForestEstimator(BaseEstimator):
    def __init__(self, max_leaf_nodes=500, l2=1,  loss='Log',
                 test_interval=100, verbose=0):
        self.max_leaf_nodes = max_leaf_nodes
        self.l2 = l2
        self.loss = loss
        self.test_interval = test_interval
        self.verbose = verbose

        self._trainer = RegularizedGreedyForest(self.param_str, self.verbose)

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

    def fit(self, X, y, sample_weight=None):
        self._trainer.fit(X, y)

    def predict(self, X):
        return self._trainer.predict(X)
