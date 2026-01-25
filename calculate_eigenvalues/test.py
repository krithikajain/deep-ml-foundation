import unittest
import numpy as np
from main import calculate_eigenvalues

class TestEigenvalues(unittest.TestCase):

    def test_example_case(self):
        """Test the specific example from the problem description."""
        matrix = [[2, 1], [1, 2]]
        expected = [3.0, 1.0]
        self.assertEqual(calculate_eigenvalues(matrix), expected)

    def test_against_numpy(self):
        """Verify against NumPy's eigenvalue calculation."""
        # We construct a symmetric matrix to ensure Real eigenvalues for this test
        m = np.random.randint(1, 10, size=(2,2))
        symmetric_matrix = (m + m.T) / 2

        #ground truth from numpy
        expected = np.linalg.eigvals(symmetric_matrix)
        expected = sorted(expected.tolist(), reverse=True)

        #our function
        actual = calculate_eigenvalues(symmetric_matrix.tolist())
        #check values using almostEqual for float precision
        np.testing.assert_almost_equal(actual, expected)

    def test_trace_det_logic(self):
        """Trace should be sum of eigenvalues, Det should be product."""
        matrix = [[4, 1], [2, 3]]
        eigenvals = calculate_eigenvalues(matrix)

        calculated_trace = sum(eigenvals)
        calculated_det = eigenvals[0] * eigenvals[1]

        actual_trace = 4 + 3
        actual_det = (4*3) - (1*2)

        self.assertAlmostEqual(calculated_trace, actual_trace)
        self.assertAlmostEqual(calculated_det, actual_det)

if __name__ == '__main__':
    unittest.main()