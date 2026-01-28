import unittest
import numpy as np
from main import calculate_matrix_mean

class TestMatrixMean(unittest.TestCase):
    
    def test_column_mean_example(self):
        """Test the specific example from the problem description."""
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        expected = [4.0, 5.0, 6.0]
        self.assertEqual(calculate_matrix_mean(matrix, 'column'), expected)

    def test_row_mean(self):
        """Test row mode against NumPy axis=1"""
        matrix = [[1, 2, 3], [4, 5, 6]]
        # Row 1 mean: 2.0, Row 2 mean: 5.0
        expected = [2.0, 5.0] 
        self.assertEqual(calculate_matrix_mean(matrix, 'row'), expected)

    def test_random_matrix(self):
        """Verify against NumPy with random data"""
        np_matrix = np.random.randint(1, 100, size=(4, 5))
        matrix_list = np_matrix.tolist()

        # Check Column Means (axis=0)
        expected_col = np.mean(np_matrix, axis=0).tolist()
        actual_col = calculate_matrix_mean(matrix_list, 'column')
        
        # Check Row Means (axis=1)
        expected_row = np.mean(np_matrix, axis=1).tolist()
        actual_row = calculate_matrix_mean(matrix_list, 'row')

        self.assertEqual(actual_col, expected_col)
        self.assertEqual(actual_row, expected_row)

if __name__ == '__main__':
    unittest.main()