import unittest
import numpy as np
from main import linear_regression_gradient_descent

class TestLinearRegression(unittest.TestCase):
    
    def test_simple_linear_function(self):
        """Test y = 1 + 2x"""
        # Features: Column of 1s (bias) and Column of inputs [1, 2, 3]
        X = np.array([
            [1, 1],
            [1, 2],
            [1, 3]
        ])
        
        # Target: y = 1 + 2x
        # x=1 -> y=3
        # x=2 -> y=5
        # x=3 -> y=7
        y = np.array([3, 5, 7])
        
        alpha = 0.1
        iterations = 1000
        
        # Expected: [Intercept=1, Slope=2]
        theta = linear_regression_gradient_descent(X, y, alpha, iterations)
        
        print(f"Learned Weights: {theta}")
        
        # Check Intercept
        self.assertAlmostEqual(theta[0], 1.0, places=2)
        # Check Slope
        self.assertAlmostEqual(theta[1], 2.0, places=2)

    def test_zero_initialization_check(self):
        """Ensure it moves away from zero"""
        X = np.array([[1, 5], [1, 2]])
        y = np.array([10, 4])
        theta = linear_regression_gradient_descent(X, y, 0.01, 100)
        self.assertFalse(np.all(theta == 0))

if __name__ == '__main__':
    unittest.main()