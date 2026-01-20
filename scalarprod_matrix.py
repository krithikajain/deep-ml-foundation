def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    """
    Multiplies each element of a matrix by a scalar value.
    Mathematically: B_ij = scalar * A_ij
    """
    result = []
    
    for row in matrix:
        new_row = []
        for element in row:
            # Scale each individual element
            new_row.append(element * scalar)
        result.append(new_row)
        
    return result

# Simple manual test (Runs only if you execute this file directly)
if __name__ == "__main__":
    test_matrix = [[1, 2], [3, 4]]
    test_scalar = 2
    print(f"Input: {test_matrix}, Scalar: {test_scalar}")
    print(f"Output: {scalar_multiply(test_matrix, test_scalar)}")