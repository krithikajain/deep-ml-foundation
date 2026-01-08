import unittest
from main import matrix_dot_vector
import numpy as np

class TestMatrixDotVector(unittest.TestCase):

    def test_using_numpy(self):
        matrix = [[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]]
        vector = [1, 0, -1]

        # .tolist() converts the numpy array back to a standard python list
        expected = np.dot(matrix, vector).tolist()

        actual = matrix_dot_vector(matrix, vector)

        self.assertEqual(actual, expected)

    def test_randomized_input(self):
        # create a random 5x4 matrix ranging from 1 to 9 and a random vector of size 4
        np_matrix = np.random.randint(1, 10, size=(5, 4))
        np_vector = np.random.randint(1, 10, size=(4,))

        # Convert to standard lists for your function
        matrix_list = np_matrix.tolist()
        vector_list = np_vector.tolist()

        expected = np.dot(np_matrix, np_vector).tolist()
        actual = matrix_dot_vector(matrix_list, vector_list)

        self.assertEqual(actual, expected)

    def test_dimension_mismatch(self):
        """
        Ensure -1 is returned when dimensions do not align.
        """
        # 2x3 Matrix
        matrix = [[1, 2, 3], [4, 5, 6]]
        # Vector of length 2 (should be 3 to match columns)
        vector = [1, 2]

        self.assertEqual(matrix_dot_vector(matrix, vector), -1)