import numpy as np


# Ref https://integratedmlai.com/system-of-equations-solution/
def zeros_matrix(rows, cols):
    A = []
    for i in range(rows):
        A.append([])
        for j in range(cols):
            A[-1].append(0.0)

    return A


def copy_matrix(M):
    rows = len(M)
    cols = len(M[0])

    MC = zeros_matrix(rows, cols)

    for i in range(rows):
        for j in range(cols):
            MC[i][j] = M[i][j]

    return MC


# The 'optimized_solve' function checks if the first element in matrix A is zero.
# If it is, the first row is swapped with a row where the first element is not zero.
# In the final step, the rows are swapped back to their original order.
def optimized_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A[0, 0] == 0:
        non_null_idx = np.argwhere(A[:, 0] != 0).flatten()[0]

        A[[0, non_null_idx]] = A[[non_null_idx, 0]]
        B[[0, non_null_idx]] = B[[non_null_idx, 0]]

    AM = copy_matrix(A)
    n = len(A)
    BM = copy_matrix(B)

    indices = list(range(n))  # allow flexible row referencing ***
    for fd in range(n):  # fd stands for focus diagonal
        fdScaler = 1.0 / AM[fd][fd]
        # FIRST: scale fd row with fd inverse.
        for j in range(n):  # Use j to indicate column looping.
            AM[fd][j] *= fdScaler
        BM[fd][0] *= fdScaler

        # SECOND: operate on all rows except fd row.
        for i in indices[0:fd] + indices[fd + 1:]:  # skip fd row.
            crScaler = AM[i][fd]  # cr stands for current row
            for j in range(n):  # cr - crScaler * fdRow.
                AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
            BM[i][0] = BM[i][0] - crScaler * BM[fd][0]

    BM = np.asarray(BM)

    return BM
