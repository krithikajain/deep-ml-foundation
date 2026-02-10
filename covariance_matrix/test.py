import unittest
import numpy as np
from main import covariance_matrix

class TestCovarianceMatrix(unittest.TestCase):
    
    def test_case_sample(self):
        """Test the specific example from the problem description."""
        vectors = [[1,2,3],[4,5,6]]
        expected = [[1.0,1.0],[1.0,1.0]]

        self.assertEqual(covariance_matrix(vectors), expected)

    def test_random_vectors(self):
        """Compare against NumPy for random data."""
        vectors = [
                    [1, 5, 6, 2, 4], 
                    [5, 2, 6, 1, 7], 
                    [8, 2, 6, 1, 3]
                ]

        #ground truth
        # np.cov expects rows=features cols=observations, matches our i/p vector
        expected = np.cov(vectors).tolist()
        actual = covariance_matrix(vectors)

        np.testing.assert_almost_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()