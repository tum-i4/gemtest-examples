import numpy as np


# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
def mul_matrix_vector(m, v):
    result = []
    for i in range(len(m[0])):
        sum_ = 0
        for j in range(len(v)):
            sum_ += m[i][j] * v[j][0]
        result.append(sum_)

    return result


def transposeMatrix(m):
    return [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]


def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]


def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant


def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors


def solve_use_inv_stackPusher(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A.tolist()
    b = b.tolist()

    A_inv = getMatrixInverse(A)
    b = mul_matrix_vector(A_inv, b)

    return np.asarray(b)
