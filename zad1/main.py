from strassen import strassen, strassen_with_count
import numpy as np
import matplotlib.pyplot as plt
from time import time


def test_for_random_2_to_power_k(k, with_count=False):
    print(f'Test for random 2^k x 2^k matrix\nk = {k}')

    n = 2 ** k
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    print(f'A:\n{A}\n')
    print(f'B:\n{B}\n')

    if with_count:
        C, count = strassen_with_count(A, B)
        print(f'Atomic operations count: {count}\n')
        print(f'C:\n{C}')
        return count
    else:
        start = time()
        C = strassen(A, B)
        print(f'C:\n{C}')
        return time() - start

    # D = np.matmul(A, B)
    # print(f'D:\n{D}\n')
    # print(f'C == D: {np.allclose(C, D)}')


def strassen_plot(max_k):
    x = np.arange(1, max_k + 1)
    y1 = [test_for_random_2_to_power_k(k) for k in x]
    y2 = [test_for_random_2_to_power_k(k, with_count=True) for k in x]

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(x, y1, 'o')
    ax.set_title("Pomiar czasu dla macierzy 2^k x 2^k")
    ax.set_xlabel("k")
    ax.set_ylabel("Czas [s]")
    ax.set_xticks(x, x)
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(x, y2, 'o')
    ax.set_title("Pomiar liczby operacji zmienno-przecinkowych dla macierzy 2^k x 2^k")
    ax.set_xlabel("k")
    ax.set_ylabel("Liczba operacji zmienno-przecinkowych")
    ax.set_xticks(x, x)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    strassen_plot(8)
