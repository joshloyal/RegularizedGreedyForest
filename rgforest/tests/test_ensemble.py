import numpy as np

import rgforest as rgf
from rgforest.tests.test_utils import get_test_data


class TestRGFTreeEnsemble(object):
    def setup(self):
        (X_train, y_train), _ = get_test_data()
        est = rgf.RegularizedGreedyForestClassifier(l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        self.ensemble = est.ensemble

    def test_smoketest(self):
        assert len(self.ensemble) == len([tree for tree in self.ensemble])
