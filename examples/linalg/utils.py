from typing import List, Tuple
from dataclasses import dataclass

import numpy as np


@dataclass
class MatrixVectorPair:
    """
    Dataclass for a pairs of matrices and vectors.
    """
    matrix: np.ndarray
    vector: np.ndarray

    def get_matrix_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.matrix, self.vector


def create_pairs(n_samples: int, matrix_max_size: Tuple[int, int] = (5, 5)) -> List[MatrixVectorPair]:
    """
    Creates a list of pairs of matrices and vectors with integers greater than or equal to zero.

    Parameters
    ----------
    n_samples : int
        Number of pairs to create.
    matrix_max_size: Tuple[int, int], optional
        The maximum size of the matrices.

    Returns
    -------
    List[MatrixVectorPair]
        A list of n pairs matrices and vectors with integers greater than or equal to zero.
    """

    # Increments each dimension by one. Ensures that the upper bounds for generating random integers are inclusive
    # since np.random.randint's high parameter is exclusive.
    matrix_max_size = tuple(map(lambda d: d+1, matrix_max_size))

    # generates random square shapes
    matrices_sizes = np.random.randint(low=[2, 2], high=matrix_max_size, size=(n_samples, 1))
    matrices_sizes = matrices_sizes.repeat(2, axis=1)
    vectors_sizes = np.asarray([[n, 1] for n in matrices_sizes[:, 1]])

    matrices: List[np.ndarray] = create_matrices(n_samples, matrices_sizes=matrices_sizes)
    vectors: List[np.ndarray] = create_vectors(n_samples, vectors_sizes=vectors_sizes)

    return [MatrixVectorPair(matrix, vector) for matrix, vector in zip(matrices, vectors)]


def create_matrices(n_samples: int,
                    matrix_max_size: Tuple[int, int] = (10, 10),
                    matrices_sizes: np.ndarray = None,
                    is_square: bool = True,
                    min_value: int = 0,
                    max_value: int = 100) -> List[np.ndarray]:
    """
    Creates a list of n matrices with integers greater than or equal to zero.

    Parameters
    ----------
    n_samples : int
        Number of matrices to create.
    matrix_max_size: Tuple[int, int], optional
        The maximum size of the matrices.
    matrices_sizes: np.ndarray, optional
        List of sizes to create specific list of mxn matrices
    is_square: bool, optional
        If set all matrices are square.
    min_value: int, optional
        Minimum value of the matrix entries.
    max_value: int, optional
        Maximum value of the matrix entries.

    Returns
    -------
    List[np.ndarray]
        A list of n matrices with integers greater than or equal to zero.
    """

    # Increments each dimension by one. Ensures that the upper bounds for generating random integers are inclusive
    # since np.random.randint's high parameter is exclusive.
    matrix_max_size = tuple(map(lambda d: d+1, matrix_max_size))

    # generates random matrices shapes (square or non square)
    if matrices_sizes is None:
        if is_square:
            matrices_sizes = np.random.randint(low=[1, 1], high=matrix_max_size, size=(n_samples, 1))
            matrices_sizes = matrices_sizes.repeat(2, axis=1)
        else:
            matrices_sizes = np.random.randint(low=[1, 1], high=matrix_max_size, size=(n_samples, 2))

    # Generates a list of random matrices using the shapes in `matrices_sizes`.
    # and increases the diagonal by 0.5 to avoid singular matrices
    return [np.random.randint(low=min_value, high=max_value, size=(m, n)) + np.eye(m, n) * 0.5 for m, n in matrices_sizes]


def create_vectors(n_samples: int,
                   vector_len: int = 10,
                   vectors_sizes: np.ndarray = None,
                   **kwargs) -> List[np.ndarray]:
    """
    Creates a list of n vectors with integers greater than or equal to zero.

    Parameters
    ----------
    n_samples : int
        Number of vectors to create.
    vector_len: int, optional
        The length of the vectors.
    vectors_sizes: np.ndarray, optional
        List of sizes to create specific list of vectors with m entries
    **kwargs : dict, optional
        min_value: int
            Minimum value of the vector entries.
        max_value: int
            Maximum value of the vector entries.

    Returns
    -------
    List[np.ndarray]
        A list of n vectors with integers greater than or equal to zero.
    """

    return create_matrices(n_samples=n_samples, matrix_max_size=(vector_len, 1), matrices_sizes=vectors_sizes,
                           is_square=False, **kwargs)
