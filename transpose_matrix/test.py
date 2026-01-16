import unittest
import numpy as np
from main import transpose_matrix

class TestTranspose(unittest.TestCase):
    
    def test_example_case(self):
        """Test the specific example from the problem description"""
        a = [[1, 2, 3], [4, 5, 6]]
        expected = [[1, 4], [2, 5], [3, 6]]
        self.assertEqual(transpose_matrix(a), expected)

    def test_random_matrix(self):
        """Compare against Numpy's official implementation"""
        # Generate random 5x3 matrix
        np_matrix = np.random.randint(1, 100, size=(5, 3))
        
        # 1. Get Ground Truth (Numpy)
        expected = np.transpose(np_matrix).tolist()
        
        # 2. Get Our Result
        # Convert input to list of lists for our function
        input_list = np_matrix.tolist()
        actual = transpose_matrix(input_list)
        
        self.assertEqual(actual, expected)

    def test_square_matrix(self):
        """Test a 2x2 square matrix"""
        a = [[1, 2], [3, 4]]
        expected = [[1, 3], [2, 4]]
        self.assertEqual(transpose_matrix(a), expected)

if __name__ == '__main__':
    unittest.main()