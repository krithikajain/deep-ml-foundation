def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    rows = len(a)
    cols = len(a[0])

    transpose = []

    for j in range(cols):
        new_row = []
        for i in range(rows):
            # Grab the element at the current column j from the current row i
            new_row.append(a[i][j])
        transpose.append(new_row)

    return transpose
