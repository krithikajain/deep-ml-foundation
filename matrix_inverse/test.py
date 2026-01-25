import unittest
import numpy as np
from main import inverse_2x2

class TestMatrixInverse(unittest.TestCase):
    
    def test_example_case(self):
        """Test the specific example from the problem description."""
        matrix = [[4, 7], [2, 6]]
        expected = [[0.6, -0.7], [-0.2, 0.4]]
        
        result = inverse_2x2(matrix)
        
        # Use assertAlmostEqual for floating point comparisons
        np.testing.assert_array_almost_equal(result, expected)

    def test_singular_matrix(self):
        """Test that a matrix with determinant 0 returns None."""
        # Determinant = (1*4) - (2*2) = 0
        matrix = [[1, 2], [2, 4]]
        self.assertIsNone(inverse_2x2(matrix))

    def test_random_invertible(self):
        """Compare against NumPy for a random matrix."""
        # Keep trying until we find a non-singular matrix
        while True:
            np_matrix = np.random.rand(2, 2)
            if np.linalg.det(np_matrix) != 0:
                break
        
        # Ground Truth
        expected = np.linalg.inv(np_matrix).tolist()
        
        # Our Result
        actual = inverse_2x2(np_matrix.tolist())
        
        np.testing.assert_array_almost_equal(actual, expected)

if __name__ == '__main__':
    unittest.main()