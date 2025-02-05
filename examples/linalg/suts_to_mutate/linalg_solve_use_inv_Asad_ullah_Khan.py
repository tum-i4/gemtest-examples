import numpy as np


def mul_matrix_vector(m, v):
    result = []
    for i in range(len(m[0])):
        sum_ = 0
        for j in range(len(v)):
            sum_ += m[i][j] * v[j][0]
        result.append(sum_)

    return result


# https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
def eliminate(r1, r2, col, target=0):
    fac = (r2[col] - target) / r1[col]
    for i in range(len(r2)):
        r2[i] -= fac * r1[i]


def gauss(a):
    for i in range(len(a)):
        if a[i][i] == 0:
            for j in range(i + 1, len(a)):
                if a[i][j] != 0:
                    a[i], a[j] = a[j], a[i]
                    break
            else:
                raise ValueError("Matrix is not invertible")
        for j in range(i + 1, len(a)):
            eliminate(a[i], a[j], i)
    for i in range(len(a) - 1, -1, -1):
        for j in range(i - 1, -1, -1):
            eliminate(a[i], a[j], i)
    for i in range(len(a)):
        eliminate(a[i], a[i], i, target=1)
    return a


def inverse(a):
    tmp = [[] for _ in a]
    for i, row in enumerate(a):
        assert len(row) == len(a)
        tmp[i].extend(row + [0] * i + [1] + [0] * (len(a) - i - 1))
    gauss(tmp)
    return [tmp[i][len(tmp[i]) // 2:] for i in range(len(tmp))]


def solve_use_inv_Asad_ullah_Khan(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    A = A.tolist()
    b = b.tolist()

    A_inv = inverse(A)
    b = mul_matrix_vector(A_inv, b)

    return np.asarray(b)
