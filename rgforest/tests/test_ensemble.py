import cPickle as pickle
import os
import tempfile

import numpy as np

import rgforest as rgf
from rgforest.tests.test_utils import get_test_data


class TestRGFTreeEnsemble(object):
    def setup(self):
        (X_train, y_train), _ = get_test_data()
        est = rgf.RegularizedGreedyForestClassifier(
            l2=0.01, max_leaf_nodes=500)
        est.fit(X_train, y_train)

        self.ensemble = est.ensemble

    def test_smoketest(self):
        n_trees = len([tree for tree in self.ensemble])
        assert len(self.ensemble) == n_trees
        assert self.ensemble.n_trees == n_trees

        tree = self.ensemble[0]
        tree.parent
        tree.node_count
        tree.n_outputs
        tree.children_left
        tree.children_right
        tree.feature
        tree.threshold
        tree.weight

    def test_tree_pickle(self):
        tree = self.ensemble[0]

        with tempfile.NamedTemporaryFile('wr') as tfile:
            file_name = tfile.name

        try:
            pickle.dump(tree, open(file_name, 'wb'))
            unpickle_tree = pickle.load(open(file_name, 'rb'))
        finally:
            os.remove(file_name)

        assert unpickle_tree.node_count == tree.node_count
        assert unpickle_tree.n_outputs == tree.n_outputs
        np.testing.assert_allclose(
            unpickle_tree.parent, tree.parent)
        np.testing.assert_allclose(
            unpickle_tree.children_left, tree.children_left)
        np.testing.assert_allclose(
            unpickle_tree.children_right, tree.children_right)
        np.testing.assert_allclose(
            unpickle_tree.feature, tree.feature)
        np.testing.assert_allclose(
            unpickle_tree.threshold, tree.threshold)
        np.testing.assert_allclose(
            unpickle_tree.weight, tree.weight)

    def test_ensemble_pickle(self):
        with tempfile.NamedTemporaryFile('wr') as tfile:
            file_name = tfile.name

        try:
            pickle.dump(self.ensemble, open(file_name, 'wb'))
            unpickle_ensemble = pickle.load(open(file_name, 'rb'))
            for unpickle_tree, tree in zip(unpickle_ensemble, self.ensemble):
                assert unpickle_tree.node_count == tree.node_count
                assert unpickle_tree.n_outputs == tree.n_outputs
                np.testing.assert_allclose(
                    unpickle_tree.parent, tree.parent)
                np.testing.assert_allclose(
                    unpickle_tree.children_left, tree.children_left)
                np.testing.assert_allclose(
                    unpickle_tree.children_right, tree.children_right)
                np.testing.assert_allclose(
                    unpickle_tree.feature, tree.feature)
                np.testing.assert_allclose(
                    unpickle_tree.threshold, tree.threshold)
                np.testing.assert_allclose(
                    unpickle_tree.weight, tree.weight)
        finally:
            os.remove(file_name)
