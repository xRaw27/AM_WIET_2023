import numpy as np
import math
from utils import split_matrix
from strassen import strassen


def inverse(A: np.ndarray):
    if A.shape == (1, 1):
        x = 1 / A[0, 0]
        return np.array([[x]]), 1

    count = [0] * 16
    A11, A12, A21, A22 = split_matrix(A)

    # A11_inv = inverse(A11)
    A11_inv, count[0] = inverse(A11)

    # S22 = A22 - A21 @ A11_inv @ A12
    S22, count[1] = strassen(A21, A11_inv)
    S22, count[2] = strassen(S22, A12)
    S22, count[3] = A22 - S22, math.prod(A.shape)

    # S22_inv = inverse(S22)
    S22_inv, count[4] = inverse(S22)

    # B11 = A11_inv @ (I + A12 @ S22_inv @ A21 @ A11_inv)
    I = np.eye(A11.shape[0])
    B11, count[5] = strassen(A12, S22_inv)
    B11, count[6] = strassen(B11, A21)
    B11, count[7] = strassen(B11, A11_inv)
    B11, count[8] = I + B11, A11.shape[0]
    B11, count[9] = strassen(A11_inv, B11)

    # B12 = -1 * (A11_inv @ A12 @ S22_inv)
    B12, count[10] = strassen(A11_inv, A12)
    B12, count[11] = strassen(B12, S22_inv)
    B12, count[12] = -1 * B12, math.prod(B12.shape)

    # B21 = -1 * (S22_inv @ A21 @ A11_inv)
    B21, count[13] = strassen(S22_inv, A21)
    B21, count[14] = strassen(B21, A11_inv)
    B21, count[15] = -1 * B21, math.prod(B21.shape)

    return np.vstack((np.hstack((B11, B12)), np.hstack((B21, S22_inv)))), sum(count)
