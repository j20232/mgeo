import unittest

import numpy as np
import mgeo


class LinalgTest(unittest.TestCase):

    def test_pca(self):
        for num, dim in [[100, 100], [100, 101], [101, 100]]:
            features = np.random.rand(num, dim)
            S, V, mean_X = mgeo.linalg.pca(features)
            num = dim if num >= dim else num
            self.assertTrue(S.shape[0] == num, "Invalid singular values")
            self.assertTrue(V.shape[0] == num, "Invalid components (num)")
            self.assertTrue(V.shape[1] == dim, "Invalid components (dim)")
            self.assertTrue(mean_X.shape[0] == dim, "Invalid mean")
