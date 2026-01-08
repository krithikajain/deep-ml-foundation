def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:

    """
    Computes the dot product of a matrix and a vector.
    
    Args:
        a: Matrix of shape (n, m)
        b: Vector of shape (m)
        
    Returns:
        List[float]: The resulting vector of shape (n)
        int: -1 if dimensions are incompatible
    """

    #dimension check -  the number of columns in the matrix equals the length of the vector.
    if len(a[0]) != len(b):
        return -1

    #initialise result
    result = []

    #compute dot product
    for row in a:
        dot_prod = 0
        
        for i in range(len(b)):
            dot_prod += row[i] * b[i]
        result.append(dot_prod)

    return result

