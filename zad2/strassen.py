import numpy as np
import math
from utils import split_matrix


def strassen(A: np.ndarray, B: np.ndarray):
    if A.shape == (1, 1):
        return A * B, 1

    A11, A12, A21, A22 = split_matrix(A)
    B11, B12, B21, B22 = split_matrix(B)

    M1, count1 = strassen(A11 + A22, B11 + B22)
    M2, count2 = strassen(A21 + A22, B11)
    M3, count3 = strassen(A11, B12 - B22)
    M4, count4 = strassen(A22, B21 - B11)
    M5, count5 = strassen(A11 + A12, B22)
    M6, count6 = strassen(A21 - A11, B11 + B12)
    M7, count7 = strassen(A12 - A22, B21 + B22)

    count = count1 + count2 + count3 + count4 + count5 + count6 + count7 + 18 * math.prod(A.shape)

    return (
        np.vstack(
            (
                np.hstack((M1 + M4 - M5 + M7, M3 + M5)),
                np.hstack((M2 + M4, M1 - M2 + M3 + M6)),
            )
        ),
        count,
    )
