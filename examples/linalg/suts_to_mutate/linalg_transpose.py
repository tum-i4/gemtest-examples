import numpy as np


def custom_transpose(matrix: np.ndarray) -> np.ndarray:
    result_matrix = np.empty(shape=list(reversed(matrix.shape)))
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            result_matrix[i, j] = matrix[j, i]

    return result_matrix
