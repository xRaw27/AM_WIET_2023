import numpy as np
from LU import LU


def det(A: np.ndarray):
    L, U, count = LU(A)
    return np.prod(np.diagonal(U)), count + A.shape[0] - 1
