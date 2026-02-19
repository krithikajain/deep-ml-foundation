import unittest
import numpy as np
from main import gradient_descent

class TestGradientDescentVariants(unittest.TestCase):
    
    def setUp(self):
        """
        Setup a simple linear problem: y = 2*x1 + 3 (x0 is bias=1)
        """
        # Features: Bias(1) and Feature(x)
        self.X = np.array([
            [1, 1],
            [1, 2],
            [1, 3],
            [1, 4]
        ])
        # Target: y = 3 + 2x
        # 1->5, 2->7, 3->9, 4->11
        self.y = np.array([5, 7, 9, 11])
        
        self.weights = np.zeros(2)
        self.lr = 0.01
        self.epochs = 3000

    def test_batch_gd(self):
        """Batch should be very stable."""
        w = gradient_descent(self.X, self.y, self.weights, self.lr, self.epochs, method='batch')
        print(f"Batch Weights: {w}")
        self.assertAlmostEqual(w[0], 3.0, places=1) # Intercept
        self.assertAlmostEqual(w[1], 2.0, places=1) # Slope

    def test_stochastic_gd(self):
        """SGD processes one by one."""
        w = gradient_descent(self.X, self.y, self.weights, self.lr, self.epochs, method='stochastic')
        print(f"Stochastic Weights: {w}")
        self.assertAlmostEqual(w[0], 3.0, places=1)
        self.assertAlmostEqual(w[1], 2.0, places=1)

    def test_mini_batch_gd(self):
        """Mini-batch processes in chunks (size 2)."""
        w = gradient_descent(self.X, self.y, self.weights, self.lr, self.epochs, batch_size=2, method='mini_batch')
        print(f"Mini-Batch Weights: {w}")
        self.assertAlmostEqual(w[0], 3.0, places=1)
        self.assertAlmostEqual(w[1], 2.0, places=1)

if __name__ == '__main__':
    unittest.main()