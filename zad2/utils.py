import numpy as np


def gen_square_matrix(k: int):
    n = 2 ** k
    A = np.random.rand(n, n)
    return A


def get_submatrix(A: np.ndarray, i: int, j: int):
    A11 = A[:i, :j]
    A12 = A[:i, j + 1:]
    A21 = A[i + 1:, :j]
    A22 = A[i + 1:, j + 1:]
    B=np.vstack((np.hstack((A11, A12)), np.hstack((A21, A22))))
    return B

