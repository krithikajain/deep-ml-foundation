import math
import numpy as np
from scipy.stats import poisson

def poisson_probability(k: int, lam: float) -> float:
    """
    Calculate the probability of observing exactly k events in a fixed interval,
    given the mean rate of events lam, using the Poisson distribution formula.
    
    Formula: (e^-lambda * lambda^k) / k!
    
    Args:
        k: Number of events (non-negative integer)
        lam: The average rate (mean) of occurrences
        
    Returns:
        Probability rounded to 5 decimal places
    """
    # 1. Calculate the numerator: e^-lam * lam^k
    numerator = math.exp(-lam) * (lam ** k)
    
    # 2. Calculate the denominator: k!
    denominator = math.factorial(k)
    
    # 3. Compute probability
    probability = numerator / denominator
    
    return round(probability, 5)

if __name__ == "__main__":
    # --- Configuration ---
    test_k = 3
    test_lam = 5
    
    print(f"Testing Poisson Probability for k={test_k}, lambda={test_lam}...")
    
    # 1. Our Manual Implementation
    my_result = poisson_probability(test_k, test_lam)
    print(f"My Implementation:   {my_result}")
    
    # 2. SciPy Implementation (The Professional Way)
    # .pmf() stands for Probability Mass Function
    scipy_result = round(poisson.pmf(test_k, test_lam), 5)
    print(f"SciPy Reference:     {scipy_result}")
    
    # 3. Verification
    if my_result == scipy_result:
        print("\n✅ SUCCESS: Calculation matches SciPy!")
    else:
        print(f"\n❌ FAILURE: Difference detected (My: {my_result} vs SciPy: {scipy_result})")