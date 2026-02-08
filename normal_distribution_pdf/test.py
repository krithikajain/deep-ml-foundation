import unittest
from scipy.stats import norm
from main import normal_pdf

class TestNormalPDF(unittest.TestCase):

    def test_example_case(self):
        """Test the specific example from the problem description."""
        x = 16
        mean = 15
        std = 2.04
        expected = 0.17342
        self.assertEqual(normal_pdf(x, mean, std), expected)

    def test_peak_at_mean(self):
        """Test that the PDF is highest at the mean."""
        mean = 0
        std = 1
        #at mean = 0, pdf is 1 /sqrt(2*pi) ~ 0.39894
        expected = 0.39894
        self.assertAlmostEqual(normal_pdf(mean, mean, std), expected, places=5)
    
    def test_against_scipy(self):
        """Scipy for random values."""
        x = 1.5
        mean = 2.0
        std = 3.0
        #ground truth
        expected = round(norm.pdf(x, loc=mean, scale=std), 5)
        actual = normal_pdf(x, mean, std)
        self.assertEqual(actual, expected)

if __name__ == '__main__':
    unittest.main()