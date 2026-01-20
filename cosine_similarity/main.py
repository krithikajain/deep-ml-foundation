import numpy as np
def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two vectors.
    
    Formula: dot(v1, v2) / ( ||v1|| * ||v2|| )
    
    Args:
        v1, v2: Numpy arrays
        
    Returns:
        float: Cosine similarity rounded to 3 decimal places
    """

    #1. Calculate Dot Product
    dot_product = np.dot(v1, v2)

    #2. Calculate Magnitudes
    # np.linalg.norm computes the Euclidean length (L2 norm)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # handle edge case if denominator is zero
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("Input vectors cannot have zero magnitude.")
    
    #3. Compute Cosine Similarity
    cosine_sim = dot_product / (norm_v1 * norm_v2)

    return round(cosine_sim, 3)