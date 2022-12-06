import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import random as random_sparse


class Node:
    def __init__(self, n, m, rank):
        self.n = n
        self.m = m
        self.rank = rank
        self.U = None
        self.Sigma = None
        self.VT = None
        self.matrix = None
        self.children = []

    def print_matrix(self, pre=""):
        if self.rank is not None:
            if self.rank > 0:
                print(pre, self.n, self.m, self.rank)
            else:
                print(pre, self.n, self.m, "zeros")
        elif self.matrix is not None:
            print(pre, self.n, self.m, "full matrix")
        else:
            print(pre, self.n, self.m, "split")
            for node in self.children:
                node.print_matrix(pre + "  ")


def split_matrix(M: np.ndarray):
    n = M.shape[0] // 2
    return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]


def compress_matrix(M, r, epsilon):
    if not np.any(M):
        return Node(*M.shape, 0)

    if min(M.shape[0], M.shape[1]) <= r + 1:
        node = Node(*M.shape, None)
        node.matrix = M
        return node

    U, Sigma, VT = svds(M, k=r + 1)

    if abs(Sigma[0]) < epsilon:
        node = Node(*M.shape, r)
        node.U = U[:, 1:]
        node.Sigma = Sigma[1:]
        node.VT = VT[1:]
    else:
        M11, M12, M21, M22 = split_matrix(M)
        node = Node(*M.shape, None)
        node.children = [
            compress_matrix(M11, r, epsilon),
            compress_matrix(M12, r, epsilon),
            compress_matrix(M21, r, epsilon),
            compress_matrix(M22, r, epsilon)
        ]

    return node


# xd = np.dot(U * Sigma, VT)
# xd[np.isclose(xd, 0)] = 0
# print(xd)


X = random_sparse(32, 32, density=0.1, random_state=0).todense()

matrix_node = compress_matrix(X, 2, 1e-6)
matrix_node.print_matrix()

