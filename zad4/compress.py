import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Node:
    def __init__(self, n, m, rank):
        self.n = n
        self.m = m
        self.rank = rank
        self.U = None
        self.Sigma = None
        self.VT = None
        self.children = []

    def print_matrix(self, pre=""):
        if self.rank is not None:
            if self.rank > 0:
                print(pre, self.n, self.m, self.rank)
            else:
                print(pre, self.n, self.m, "zeros")
        else:
            print(pre, self.n, self.m, "split,   children: ", len(self.children))
            for node in self.children:
                node.print_matrix(pre + "  ")

    def __get_matrix_to_draw(self):
        if self.rank is not None:
            if self.rank > 0:
                m = np.zeros((self.n, self.m))
                m[:, :self.rank] = 1
                m[:self.rank, :] = 1
                return m
            else:
                return np.zeros((self.n, self.m))
        else:
            return np.vstack(
                (
                    np.hstack((self.children[0].__get_matrix_to_draw(), self.children[1].__get_matrix_to_draw())),
                    np.hstack((self.children[2].__get_matrix_to_draw(), self.children[3].__get_matrix_to_draw())),
                )
            )

    def draw_matrix(self):
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.matshow(self.__get_matrix_to_draw(), cmap=ListedColormap(['w', 'k']))
        plt.show()


def split_matrix(M: np.ndarray):
    n = M.shape[0] // 2
    return M[:n, :n], M[:n, n:], M[n:, :n], M[n:, n:]


def compress_matrix(M, r, epsilon):
    if not np.any(M):
        return Node(*M.shape, 0)

    if min(M.shape[0], M.shape[1]) <= r + 1:
        U, Sigma, VT = np.linalg.svd(M)
        node = Node(*M.shape, M.shape[0])
        node.U = np.array(U)
        node.Sigma = Sigma
        node.VT = np.array(VT)
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


def decompress(node):
    if node.rank is not None:
        if node.rank > 0:
            return node.U * node.Sigma @ node.VT
        else:
            return np.zeros((node.n, node.m))
    else:
        return np.vstack(
            (
                np.hstack((decompress(node.children[0]), decompress(node.children[1]))),
                np.hstack((decompress(node.children[2]), decompress(node.children[3]))),
            )
        )
