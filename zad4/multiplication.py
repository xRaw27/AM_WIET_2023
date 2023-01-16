from compress import Node, compress_matrix, decompress
import numpy as np
from scipy.sparse import random as random_sparse


def split(X: np.array):
    n = X.shape[0] // 2
    return X[:n], X[n:]


def split_horizontal(X: np.ndarray):
    n = X.shape[0] // 2
    return X[:n, :], X[n:, :]


def split_vertical(X: np.ndarray):
    n = X.shape[1] // 2
    return X[:, :n], X[:, n:]


def split_compressed(v: Node):
    n, m = v.n // 2, v.m // 2
    nodes = [Node(n, m, v.rank), Node(n, m, v.rank), Node(n, m, v.rank), Node(n, m, v.rank)]
    U1, U2 = split_horizontal(v.U)
    VT1, VT2 = split_vertical(v.VT)
    nodes[0].U = U1
    nodes[0].Sigma = v.Sigma
    nodes[0].VT = VT1
    nodes[1].U = U1
    nodes[1].Sigma = v.Sigma
    nodes[1].VT = VT2
    nodes[2].U = U2
    nodes[2].Sigma = v.Sigma
    nodes[2].VT = VT1
    nodes[3].U = U2
    nodes[3].Sigma = v.Sigma
    nodes[3].VT = VT2
    return nodes


def matrix_vector_mult(v: Node, X: np.ndarray):
    if len(v.children) == 0:
        if v.rank > 0:
            return v.U * v.Sigma @ v.VT @ X
        else:
            return np.zeros(X.shape)

    X1, X2 = split_horizontal(X)
    Y1 = matrix_vector_mult(v.children[0], X1)
    Y2 = matrix_vector_mult(v.children[1], X2)
    Y3 = matrix_vector_mult(v.children[2], X1)
    Y4 = matrix_vector_mult(v.children[3], X2)
    return np.vstack((Y1 + Y2, Y3 + Y4))


def recompress(A, B, epsilon):
    Qa, Ra = np.linalg.qr(A, mode="reduced")
    Qb, Rb = np.linalg.qr(B.T, mode="reduced")

    U, Sigma, VT = np.linalg.svd(Ra @ Rb.T)

    for r in range(0, Sigma.shape[0]):
        if Sigma[r] < epsilon:
            return Qa @ U[:, :r], Sigma[:r], (Qb @ VT.T[:, :r]).T, r

    return Qa @ U, Sigma, (Qb @ VT.T).T, Sigma.shape[0]


def addition(v: Node, w: Node, epsilon):
    U = np.hstack((v.U, w.U))
    Sigma = np.hstack((v.Sigma, w.Sigma))
    VT = np.vstack((v.VT, w.VT))
    U, Sigma, VT, r = recompress(U * Sigma, VT, epsilon=epsilon)
    node = Node(v.n, v.m, r)
    node.U = U
    node.Sigma = Sigma
    node.VT = VT
    return node


def matrix_matrix_add(v: Node, w: Node, epsilon):
    if len(v.children) == 0 and len(w.children) == 0:
        if v.rank == 0 and w.rank == 0:
            return Node(v.n, v.m, 0)
        elif v.rank == 0:
            return w
        elif w.rank == 0:
            return v
        else:
            return addition(v, w, epsilon=epsilon)
    elif len(v.children) > 0 and len(w.children) > 0:
        node = Node(v.n, v.m, None)
        node.children = [
            matrix_matrix_add(v.children[0], w.children[0], epsilon=epsilon),
            matrix_matrix_add(v.children[1], w.children[1], epsilon=epsilon),
            matrix_matrix_add(v.children[2], w.children[2], epsilon=epsilon),
            matrix_matrix_add(v.children[3], w.children[3], epsilon=epsilon)
        ]
        return node
    elif len(v.children) == 0 and len(w.children) > 0:
        if v.rank == 0:
            return w
        nodes = split_compressed(v)
        node = Node(v.n, v.m, None)
        node.children = [
            matrix_matrix_add(nodes[0], w.children[0], epsilon=epsilon),
            matrix_matrix_add(nodes[1], w.children[1], epsilon=epsilon),
            matrix_matrix_add(nodes[2], w.children[2], epsilon=epsilon),
            matrix_matrix_add(nodes[3], w.children[3], epsilon=epsilon)
        ]
        return node
    else:
        if w.rank == 0:
            return v
        nodes = split_compressed(w)
        node = Node(v.n, v.m, None)
        node.children = [
            matrix_matrix_add(v.children[0], nodes[0], epsilon=epsilon),
            matrix_matrix_add(v.children[1], nodes[1], epsilon=epsilon),
            matrix_matrix_add(v.children[2], nodes[2], epsilon=epsilon),
            matrix_matrix_add(v.children[3], nodes[3], epsilon=epsilon)
        ]
        return node


def multiply_recursive(v: Node, w: Node, epsilon):
    A = v.children
    B = w.children
    if len(A) == 0:
        A = split_compressed(v)
    if len(B) == 0:
        B = split_compressed(w)

    node = Node(v.n, v.m, None)
    node.children = [
        matrix_matrix_add(matrix_matrix_mult(A[0], B[0], epsilon), matrix_matrix_mult(A[1], B[2], epsilon), epsilon),
        matrix_matrix_add(matrix_matrix_mult(A[0], B[1], epsilon), matrix_matrix_mult(A[1], B[3], epsilon), epsilon),
        matrix_matrix_add(matrix_matrix_mult(A[2], B[0], epsilon), matrix_matrix_mult(A[3], B[2], epsilon), epsilon),
        matrix_matrix_add(matrix_matrix_mult(A[2], B[1], epsilon), matrix_matrix_mult(A[3], B[3], epsilon), epsilon)
    ]
    return node


def matrix_matrix_mult(v: Node, w: Node, epsilon):
    if len(v.children) == 0 and len(w.children) == 0:
        if v.rank == 0 or w.rank == 0:
            return Node(v.n, v.m, 0)
        else:
            node = Node(v.n, v.m, rank=v.rank)
            node.U = v.U
            node.Sigma = v.Sigma
            node.VT = (v.VT @ w.U * w.Sigma) @ w.VT
            return node
    if len(v.children) > 0 and len(w.children) > 0:
        return multiply_recursive(v, w, epsilon=epsilon)
    if len(v.children) == 0 and len(w.children) > 0:
        if v.rank == 0:
            return Node(v.n, v.m, 0)
        return multiply_recursive(v, w, epsilon=epsilon)
    if len(v.children) > 0 and len(w.children) == 0:
        if w.rank == 0:
            return Node(w.n, w.m, 0)
        return multiply_recursive(v, w, epsilon=epsilon)


def generate_matrix_compressed(n):
    M = random_sparse(n, n, density=0.01).todense()
    m = compress_matrix(M, r=2, epsilon=0.001)
    return M, m


def test_matrix_vector_mult():
    M1, m1 = generate_matrix_compressed(256)

    m1.draw_matrix()
    x = np.random.random((256, 1))
    y = matrix_vector_mult(m1, x)

    print(f"||y - M1 @ x||^2 = {np.sum(np.square(y - (M1 @ x)))}")


def test_matrix_matrix_add():
    M1, m1 = generate_matrix_compressed(256)
    M2, m2 = generate_matrix_compressed(256)
    m3 = matrix_matrix_add(m1, m2, epsilon=0.001)

    m1.draw_matrix()
    m2.draw_matrix()
    m3.draw_matrix()

    M3 = decompress(m3)
    print(f"||M3 - (M1 + M2)||^2 = {np.sum(np.square(M3 - (M1 + M2)))}")


def test_matrix_matrix_mult():
    M = random_sparse(256, 256, density=0.01).todense()
    m1 = compress_matrix(M, r=2, epsilon=0.001)
    m2 = compress_matrix(M, r=2, epsilon=0.001)
    m3 = matrix_matrix_mult(m1, m2, epsilon=0.001)

    m1.draw_matrix()
    m3.draw_matrix()

    M3 = decompress(m3)
    print(f"||M3 - (M1 @ M1)||^2 = {np.sum(np.square(M3 - (M @ M)))}")

