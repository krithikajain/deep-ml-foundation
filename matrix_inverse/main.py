def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.
    
    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]
    
    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)

    Formula:(1/det) * [[d,-b], [-c,a]]
    """
    #1. extract the values
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]

    #2. calc determinant
    det = (a * d) - (c * b)

    #check if singular matrix
    if det == 0:
        return None

    #apply inverse formula
    inv_det = 1 / det
    inverse = [
        [d * inv_det, -b * inv_det],
        [-c * inv_det, a * inv_det]]

    return inverse
    