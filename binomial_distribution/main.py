import math

def binomial_probability(n: int, k: int, p: float) -> float:
    """
    Calculate the probability of exactly k successes in n Bernoulli trials.
    
    Args:
        n: Total number of trials
        k: Number of successes
        p: Probability of success on each trial
    
    Returns:
        Probability of k successes

    Formula: C(n,k) * p^k * (1-p)^(n-k)
             C(n,k) = n! / (k! * (n-k)!)
    """

    #calculations
    combinations = math.comb(n, k)
    prob_success = p ** k
    prob_failure = (1 - p) ** (n - k)

    result = combinations * prob_success * prob_failure
    return round(result, 5)