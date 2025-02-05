# The metamorphic relations (mr) originate from:
# Saha, Prashanta, and Upulee Kanewala.
# Using Metamorphic Relations to Improve The Effectiveness of Automatically Generated Test Cases.
# 2022 IEEE/ACIS 20th International Conference on Software Engineering Research, Management and Applications (SERA).
# IEEE, 2022.


import numpy as np
import gemtest as gmt

from utils import create_pairs, MatrixVectorPair
from suts_to_mutate import original_solve, optimized_solve, solve_use_inv_stackPusher, solve_use_inv_Asad_ullah_Khan

np.seterr(all='raise')
np.random.seed(7)

_NUM_OF_SAMPLES = 100

mr11 = gmt.create_metamorphic_relation(name='M11', data=create_pairs(_NUM_OF_SAMPLES))
mr12 = gmt.create_metamorphic_relation(name='M12', data=create_pairs(_NUM_OF_SAMPLES))
mr13 = gmt.create_metamorphic_relation(name='M13', data=create_pairs(_NUM_OF_SAMPLES))
mr14 = gmt.create_metamorphic_relation(name='M14', data=create_pairs(_NUM_OF_SAMPLES))
mr15 = gmt.create_metamorphic_relation(name='M15', data=create_pairs(_NUM_OF_SAMPLES))


@gmt.transformation(mr11)
def multiplication(source_input: MatrixVectorPair) -> MatrixVectorPair:
    matrix, vector = source_input.get_matrix_vector()
    b = np.random.randint(low=1, high=100, size=1)

    return MatrixVectorPair(matrix * b, vector * b)


@gmt.transformation(mr12)
def permute_row_elements(source_input: MatrixVectorPair) -> MatrixVectorPair:
    matrix, vector = source_input.get_matrix_vector()
    indices = np.arange(vector.shape[0])
    np.random.shuffle(indices)

    return MatrixVectorPair(matrix[indices], vector[indices])


@gmt.transformation(mr13)
def matrix_vector_addition(source_input: MatrixVectorPair) -> MatrixVectorPair:
    matrix, vector = source_input.get_matrix_vector()

    matrix = np.add(matrix, matrix)
    vector = np.add(vector, vector)

    return MatrixVectorPair(matrix, vector)


@gmt.transformation(mr14)
def multiplication_with_transpose_matrix(source_input: MatrixVectorPair) -> MatrixVectorPair:
    matrix, vector = source_input.get_matrix_vector()

    vector = np.matmul(matrix.T, vector)
    matrix = np.matmul(matrix.T, matrix)

    return MatrixVectorPair(matrix, vector)


@gmt.transformation(mr15)
def multiplication_with_identity_matrix(source_input: MatrixVectorPair) -> MatrixVectorPair:
    matrix, vector = source_input.get_matrix_vector()

    identity_matrix = np.identity(matrix.shape[0])
    matrix = np.matmul(matrix, identity_matrix)

    return MatrixVectorPair(matrix, vector)


@gmt.relation(mr11, mr12, mr13, mr14, mr15)
def equal_sum_relation(source_input: np.ndarray, followup_output: np.ndarray) -> bool:
    return bool(np.isclose(source_input.sum(), followup_output.sum()))


@gmt.system_under_test(mr11, mr12, mr13, mr14, mr15)
def test_numpy_solve(source_input: MatrixVectorPair) -> np.ndarray:
    matrix, vector = source_input.get_matrix_vector()

    return np.linalg.solve(matrix, vector)


@gmt.system_under_test(mr11, mr12, mr13, mr14, mr15)
def test_pure_python_original_solve(source_input: MatrixVectorPair) -> np.ndarray:
    matrix, vector = source_input.get_matrix_vector()

    return original_solve(matrix, vector)


@gmt.system_under_test(mr11, mr12, mr13, mr14, mr15)
def test_pure_python_optimized_solve(source_input: MatrixVectorPair) -> np.ndarray:
    matrix, vector = source_input.get_matrix_vector()

    return optimized_solve(matrix, vector)


@gmt.system_under_test(mr11, mr12, mr13, mr14, mr15)
def test_pure_python_solve_stackPusher(source_input: MatrixVectorPair) -> np.ndarray:
    matrix, vector = source_input.get_matrix_vector()

    return solve_use_inv_stackPusher(matrix, vector)


@gmt.system_under_test(mr11, mr12, mr13, mr14, mr15)
def test_pure_python_solve_Asad_ullah_Khan(source_input: MatrixVectorPair) -> np.ndarray:
    matrix, vector = source_input.get_matrix_vector()

    return solve_use_inv_Asad_ullah_Khan(matrix, vector)
