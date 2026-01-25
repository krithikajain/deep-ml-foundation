def calculate_eigenvalues(matrix: list[lsit[float]]) -> list[float]:
    """
    Calculates the eigenvalues of a 2x2 matrix using the characteristic equation.
    
    Equation: lambda^2 - trace*lambda + determinant = 0
    Solved via Quadratic Formula.
    
    Args:
        matrix: A 2x2 list of lists [[a, b], [c, d]]
        
    Returns:
        List of eigenvalues sorted from highest to lowest.
    """
    #1. Extract matrix elements
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    #2. calc trace and determinant
    trace = a + d
    determinant = a * d - b * c

    #3. Solve characteristic equation using Quadratic Formula
    # determinant is b^2 - 4ac where a=1, b=-trace, c=determinant
    discriminant = trace**2 - 4 * determinant

    #solving quadratic equation -b ± √(b² - 4ac) / 2a 
    if discriminant < 0:
        pass #negative roots

    #real roots
    eigenvalue1 = (trace + discriminant**0.5) / 2
    eigenvalue2 = (trace - discriminant**0.5) / 2

    return sorted([eigenvalue1, eigenvalue2], reverse=True)

    

