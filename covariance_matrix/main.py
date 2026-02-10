def covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    """
    Calculate the covariance matrix for a given set of vectors.
    
    Args:
        vectors: A list of lists, where each inner list is a feature 
                 (e.g., [feature_1_data, feature_2_data, ...])
    
    Returns:
        The covariance matrix as a list of lists.
    """
    n_features = len(vectors)
    n_observations = len(vectors[0])
    
    # 1. Calculate means for each feature
    means = [sum(feature) / n_observations for feature in vectors]
    
    # 2. Initialize the covariance matrix
    cov_matrix = [[0.0] * n_features for _ in range(n_features)]
    
    # 3. Calculate Covariance for each pair (i, j)
    for i in range(n_features):
        for j in range(n_features):
            covariance = 0.0
            for k in range(n_observations):
                # (x_k - x_mean) * (y_k - y_mean)
                diff_i = vectors[i][k] - means[i]
                diff_j = vectors[j][k] - means[j]
                covariance += diff_i * diff_j
            
            # Divide by (N-1) for Sample Covariance
            # (If the problem implies Population Covariance, use N)
            cov_matrix[i][j] = covariance / (n_observations - 1)
            
    return cov_matrix