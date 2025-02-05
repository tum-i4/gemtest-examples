# The metamorphic relations (mr) originate from:
# Saha, Prashanta, and Upulee Kanewala.
# Using Metamorphic Relations to Improve The Effectiveness of Automatically Generated Test Cases.
# 2022 IEEE/ACIS 20th International Conference on Software Engineering Research, Management and Applications (SERA).
# IEEE, 2022.

import numpy as np
import gemtest as gmt

from utils import create_matrices
from suts_to_mutate import custom_transpose

np.seterr(all='raise')
np.random.seed(7)

_NUM_OF_SAMPLES = 100

mr1 = gmt.create_metamorphic_relation(name='MR1', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr2 = gmt.create_metamorphic_relation(name='MR2', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr3 = gmt.create_metamorphic_relation(name='MR3', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr4 = gmt.create_metamorphic_relation(name='MR4', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr5 = gmt.create_metamorphic_relation(name='MR5', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr6 = gmt.create_metamorphic_relation(name='MR6', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr7 = gmt.create_metamorphic_relation(name='MR7', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr8 = gmt.create_metamorphic_relation(name='MR8', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr9 = gmt.create_metamorphic_relation(name='MR9', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))
mr10 = gmt.create_metamorphic_relation(name='MR10', data=create_matrices(_NUM_OF_SAMPLES, is_square=False))


@gmt.transformation(mr1)
def scalar_addition(source_input: np.ndarray) -> np.ndarray:
    b = np.random.randint(low=0, high=100, size=1)
    return np.add(source_input, b)


@gmt.transformation(mr2)
def addition_with_identity_matrix(source_input: np.ndarray) -> np.ndarray:
    if source_input.shape[0] > source_input.shape[1]:
        identity_matrix = np.identity(source_input.shape[0])
        identity_matrix = identity_matrix[:, :source_input.shape[1]]
    elif source_input.shape[0] < source_input.shape[1]:
        identity_matrix = np.identity(source_input.shape[1])
        identity_matrix = identity_matrix[:source_input.shape[0], :]
    else:
        identity_matrix = np.identity(source_input.shape[0])

    return np.add(source_input, identity_matrix)


@gmt.transformation(mr3)
def scalar_multiplication(source_input: np.ndarray) -> np.ndarray:
    b = np.random.randint(low=1, high=100, size=1)
    return np.multiply(source_input, b)


@gmt.transformation(mr4)
def multiplication_with_identity_matrix(source_input: np.ndarray) -> np.ndarray:
    identity_matrix = np.identity(source_input.shape[1])
    return np.matmul(source_input, identity_matrix)


@gmt.transformation(mr5)
def transpose(source_input: np.ndarray) -> np.ndarray:
    return source_input.T


@gmt.transformation(mr6)
def matrix_addition(source_input: np.ndarray) -> np.ndarray:
    return np.add(source_input, source_input)


@gmt.transformation(mr7)
def matrix_multiplication(source_input: np.ndarray) -> np.ndarray:
    return np.multiply(source_input, source_input)


@gmt.transformation(mr8)
def permute_column(source_input: np.ndarray) -> np.ndarray:
    indices = np.arange(source_input.shape[1])
    np.random.shuffle(indices)

    return source_input[:, indices]


@gmt.transformation(mr9)
def permute_row(source_input: np.ndarray) -> np.ndarray:
    indices = np.arange(source_input.shape[0])
    np.random.shuffle(indices)

    return source_input[indices]


@gmt.transformation(mr10)
def permute_element(source_input: np.ndarray) -> np.ndarray:
    tmp_input = source_input.copy()

    for i in range(tmp_input.shape[0]):
        np.random.shuffle(tmp_input[i])

    np.random.shuffle(tmp_input)

    return tmp_input


@gmt.relation(mr4, mr5, mr8, mr9, mr10)
def equal_sum_relation(source_input: np.ndarray, followup_output: np.ndarray) -> bool:
    return bool(np.isclose(source_input.flatten().sum(), followup_output.flatten().sum()))


@gmt.relation(mr1, mr2, mr3, mr6, mr7)
def greater_equal_sum_relation(source_input: np.ndarray, followup_output: np.ndarray) -> bool:
    return bool(np.greater_equal(followup_output.flatten().sum(), source_input.flatten().sum()))


@gmt.system_under_test(mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mr10)
def test_numpy_transpose(input_matrix: np.ndarray) -> np.ndarray:
    return input_matrix.T


@gmt.system_under_test(mr1, mr2, mr3, mr4, mr5, mr6, mr7, mr8, mr9, mr10)
def test_pure_python_transpose(input_matrix: np.ndarray) -> np.ndarray:
    return custom_transpose(input_matrix)
