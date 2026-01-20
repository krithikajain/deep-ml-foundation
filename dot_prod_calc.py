import numpy as np
	# """
	# Calculate the dot product of two vectors.
	# Args:
	# 	vec1 (numpy.ndarray): 1D array representing the first vector.
	# 	vec2 (numpy.ndarray): 1D array representing the second vector.
	# """
def calculate_dot_product(vec1, vec2) -> float:
    return np.dot(vec1, vec2)

if __name__ == "__main__":
    vec_a = np.array([1, 2, 3])
    vec_b = np.array([4, 5, 6])
    result = calculate_dot_product(vec_a, vec_b)
    print("Dot Product:", result)