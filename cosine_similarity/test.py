import unittest
import numpy as np
from main import cosine_similarity

class TestCosineSimilarity(unittest.TestCase):

    def test_perfect_similarity(self):
        """Vectors pointing in same direction should be 1.0"""
        v1 = np.array([1, 2, 3])
        v2 = np.array([2, 4, 6]) # Same direction, just longer
        self.assertEqual(cosine_similarity(v1, v2), 1.0)

    def test_orthogonal(self):
        """Vectors at 90 degrees should be 0.0"""
        # [1, 0] is on X-axis, [0, 1] is on Y-axis
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])
        self.assertEqual(cosine_similarity(v1, v2), 0.0)

    def test_opposite(self):
        """Vectors pointing in opposite directions should be -1.0"""
        v1 = np.array([1, 1])
        v2 = np.array([-1, -1])
        self.assertEqual(cosine_similarity(v1, v2), -1.0)

if __name__ == '__main__':
    unittest.main()