import numpy as np

def gradient_descent(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_epochs: int, batch_size: int = 1, method: str = 'batch') -> np.ndarray:
    """
    Performs gradient descent variants: Batch, Stochastic, and Mini-Batch.
    
    Args:
        X: Feature matrix (m samples, n features)
        y: Target vector (m samples)
        weights: Initial weights (n features)
        learning_rate: Alpha
        n_epochs: Number of passes through the entire dataset
        batch_size: Size of batch (only used for mini_batch)
        method: 'batch', 'stochastic', or 'mini_batch'
        
    Returns:
        Final weights
    """
    m, n = X.shape
    
    # Copy weights to avoid modifying original array
    w = weights.copy()
    
    # 1. Determine effective batch size based on method
    if method == 'batch':
        bs = m
    elif method == 'stochastic':
        bs = 1
    elif method == 'mini_batch':
        bs = batch_size
    else:
        raise ValueError("Method must be 'batch', 'stochastic', or 'mini_batch'")
    
    # 2. Training Loop
    for epoch in range(n_epochs):
        # Iterate through dataset in chunks of size 'bs'
        # Range(start, stop, step) handles the "slicing" logic
        for i in range(0, m, bs):
            # Create the batch (Slicing X and y)
            X_batch = X[i : i + bs]
            y_batch = y[i : i + bs]
            
            # Actual size of this batch (last batch might be smaller)
            current_batch_size = X_batch.shape[0]
            
            # --- Standard Gradient Descent Step ---
            
            # A. Prediction: X * w
            predictions = X_batch @ w
            
            # B. Error: Pred - Actual
            error = predictions - y_batch
            
            # C. Gradient: (2/m) * X.T * Error
            # Note: Using 2/m matches the derivative of raw MSE.
            # If using 1/2 MSE, use 1/m. Let's stick to standard 2/m for MSE derivative.
            gradient = (2 / current_batch_size) * (X_batch.T @ error)
            
            # D. Update
            w = w - (learning_rate * gradient)
            
    return w