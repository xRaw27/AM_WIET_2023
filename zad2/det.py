import numpy as np
from utils import get_submatrix


def det(A: np.ndarray):
    if A.shape == (1, 1):
        return A[0, 0], 0

    result = 0
    count = 0

    for j in range(A.shape[1]):
        submatrix = get_submatrix(A, 0, j)
        submatrix_det, submatrix_count = det(submatrix)
        result += ((-1) ** j) * A[0, j] * submatrix_det
        count += submatrix_count + 3

    return result, count
