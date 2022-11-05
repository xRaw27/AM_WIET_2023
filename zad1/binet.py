import numpy as np
import math


def split_matrix(M: np.ndarray):
    n = M.shape[0] // 2
    return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]


def binet(A: np.ndarray, B: np.ndarray):
    if A.shape == (1, 1):
        return A * B

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    C1 = binet(A11, B11) + binet(A12, B21)
    C2 = binet(A11, B12) + binet(A12, B22)
    C3 = binet(A21, B11) + binet(A22, B21)
    C4 = binet(A21, B12) + binet(A22, B22)

    return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4))))


def binet_with_count(A: np.ndarray, B: np.ndarray):
    if A.shape == (1, 1):
        return A * B, 1

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    C11a, count1a = binet_with_count(A11, B11)
    C11b, count1b = binet_with_count(A12, B21)
    C12a, count2a = binet_with_count(A11, B12)
    C12b, count2b = binet_with_count(A12, B22)
    C21a, count3a = binet_with_count(A21, B11)
    C21b, count3b = binet_with_count(A22, B21)
    C22a, count4a = binet_with_count(A21, B12)
    C22b, count4b = binet_with_count(A22, B22)
    C1 = C11a + C11b
    C2 = C12a + C12b
    C3 = C21a + C21b
    C4 = C22a + C22b

    count = (
        count1a
        + count1b
        + count2a
        + count2b
        + count3a
        + count3b
        + count4a
        + count4b
        + 4 * math.prod(A.shape)
    )

    return np.vstack((np.hstack((C1, C2)), np.hstack((C3, C4)))), count
