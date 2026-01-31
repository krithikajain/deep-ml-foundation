import unittest
from scipy.stats import binom
from main import binomial_probability

class TestBinomial(unittest.TestCase):

    def test_example_case(self):
        """specific ex from problem description"""
        n, k, p = 6, 2, 0.5
        expected = 0.23438
        self.assertEqual(binomial_probability(n, k, p), expected)

    def test_against_scipy(self):
        """verify against scipy for random parameters"""
        n, k, p = 10, 3, 0.3
        #Ground truth from scipy
        # .pmf Probability Mass Function
        expected = round(binom.pmf(k, n, p), 5)
        actual = binomial_probability(n, k, p)
        self.assertEqual(actual, expected)

    def test_edge_cases(self):
        """test for 0 and all successes"""
        n, p = 5, 0.5
        #0 successes(
        self.assertEqual(binomial_probability(n, 0, p), 0.03125)
        #all successes
        self.assertEqual(binomial_probability(n, n, p), 0.03125)

if __name__ == "__main__":
    unittest.main()