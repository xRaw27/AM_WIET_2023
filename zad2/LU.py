import numpy as np
import math
from utils import split_matrix
from strassen import strassen
from inverse import inverse


def LU(A: np.ndarray):
    if A.shape == (1, 1):
        return np.array([[1]]), A, 0

    count = [0] * 10
    A11, A12, A21, A22 = split_matrix(A)

    L11, U11, count[0] = LU(A11)
    L11_inv, count[1] = inverse(L11)
    U11_inv, count[2] = inverse(U11)

    L21, count[3] = strassen(A21, U11_inv)
    U12, count[4] = strassen(L11_inv, A12)

    S, count[5] = strassen(A21, U11_inv)
    S, count[6] = strassen(S, L11_inv)
    S, count[7] = strassen(S, A12)
    S, count[8] = A22 - S, math.prod(S.shape)

    L22, U22, count[9] = LU(S)

    return (
        np.vstack((np.hstack((L11, np.zeros_like(A12))), np.hstack((L21, L22)))),
        np.vstack((np.hstack((U11, U12)), np.hstack((np.zeros_like(A21), U22)))),
        sum(count)
    )
