import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> np.ndarray:
    """
        Perform linear regression using gradient descent.
        
        Args:
            X: Feature matrix (m samples, n features). First col should be 1s.
            y: Target vector (m samples).
            alpha: Learning rate.
            iterations: Number of update steps.
        Formula:
            
        Returns:
            Final weights as a 1D array.
    """
    m, n = X.shape
    #1. reshape y to be a column vector to match dimensions  
    y = y.reshape(-1, 1) 

    #2. Initialize weights to zeros theta
    theta = np.zeros((n,1))

    #3. Gradient descent loop
    for _ in range(iterations):
        #make predictions: h_tetha(x) = X @ theta
        predictions = X @ theta #shape (m,1)

        #calc error: actual - predicted
        error = predictions - y #shape (m,1)

        #calc gradient: (1/m) * X^T @ error
        gradient = (1/m) * (X.T @ error) #shape (n,1)

        #updating weights
        theta -= learning_rate * gradient

    return theta.flatten() #return as 1D array


