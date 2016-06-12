import numpy as np
from rgforest import dataset

module_rng = np.random.RandomState(1234)


class TestRGFMatrix(object):
    """Test suite of RGFMatrix"""
    def test_dataset_ones_wide(self):
        n_features = 100
        n_samples = 10

        X = np.ones((n_samples, n_features)).astype(np.float64)
        y = module_rng.choice([1, -1], n_samples).astype(np.float64)
        rgf_matrix = dataset.RGFMatrix(X, y)
        del X
        del y

        assert rgf_matrix.n_features == n_features
        assert rgf_matrix.n_samples == n_samples

        np.testing.assert_allclose(
            rgf_matrix.sum_features(), n_features * n_samples)

    def test_dataset_ones_long(self):
        n_features = 10
        n_samples = 100

        X = np.ones((n_samples, n_features)).astype(np.float64)
        y = module_rng.choice([1, -1], n_samples).astype(np.float64)
        rgf_matrix = dataset.RGFMatrix(X, y)
        del X
        del y

        assert rgf_matrix.n_features == n_features
        assert rgf_matrix.n_samples == n_samples

        np.testing.assert_allclose(
            rgf_matrix.sum_features(), n_features * n_samples)

    def test_dataset_ones_equal(self):
        n_features = 10
        n_samples = 10

        X = np.ones((n_samples, n_features)).astype(np.float64)
        y = module_rng.choice([1, -1], n_samples).astype(np.float64)
        rgf_matrix = dataset.RGFMatrix(X, y)
        del X
        del y

        assert rgf_matrix.n_features == n_features
        assert rgf_matrix.n_samples == n_samples

        np.testing.assert_allclose(
            rgf_matrix.sum_features(), n_features * n_samples)

    def test_dataset_random(self):
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
                np.testing.assert_allclose(
                    rgf_matrix.feature_get(i, j), X[i, j])
