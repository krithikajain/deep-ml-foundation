def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    """
    Calculates the mean of a matrix either by row or by column.
    
    Args:
        matrix: List of lists containing numbers
        mode: 'row' or 'column'
        
    Returns:
        List of means
    """
    if not matrix:
        return []

    rows = len(matrix)
    cols = len(matrix[0])
    means = []

    if mode == "column":
        #iterate over every column first 
        for j in range(cols):
            col_sum = 0
            for i in range(rows):
                col_sum += matrix[i][j]
            means.append(col_sum / rows)

    elif mode == "row":
        # iterate over every row first
        for row in matrix:
            row_sum = sum(row)
            means.append(row_sum / cols)
        
    else:
        raise ValueError("Mode must be 'row' or 'column'")

    return means