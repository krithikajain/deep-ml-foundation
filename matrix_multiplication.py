import numpy as np
def matrixmul(a: list[list[int|float]], b: list[list[int|float]]) -> list[list[int|float]]:
    """
    Multiplies two matrices A and B.
    Formula: C_ij = Sum(A_ik * B_kj)
    
    Returns:
        The result matrix or -1 if dimensions mismatch.
    """

    # dimension check cols A must be equal to rows B
    if len(a[0]) != len(b):
        return -1
    
    rows_a = len(a)
    cols_a = len(a[0]) #same as rows_b
    cols_b = len(b[0])

    #initialize result with zeros and shape (rows_a x cols_b)
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]

    # three loops where it iterates row_a -> cols_b -> result
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result


#verifying my function with numpy
if __name__ == "__main__":
    A = [[1,2], [2,4]]
    B = [[2,1], [3,4]]

    my_result = matrixmul(A, B)
    np_result = np.matmul(A, B).tolist()

    if my_result == np_result:
        print("Test passed! results match.")
    else:
        print("Test failed! results do not match.")
        print(f"My result: {my_result}")
        print(f"Numpy result: {np_result}")

#test with incompatible dimensions
    C = [[1,2,3], [4,5,6]]
    D = [[1,2], [3,4]]
    print(f"Testing Mismatch: {len(C[0])} cols vs {len(D)} rows")
    print(f"Result: {matrixmul(C, D)} (Expected: -1)")