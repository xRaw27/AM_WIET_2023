import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from scipy.optimize import curve_fit
from scipy.sparse import random as random_sparse
from multiplication import matrix_vector_mult, matrix_matrix_add, matrix_matrix_mult

from compress import compress_matrix


def test_matrix_vector_mult(max_k, density=0.1, r=2, epsilon=0.001):
    times = []

    for k in range(1, max_k + 1):
        print(f"### matrix_vector_mult 2^{k} x 2^{k} ###")
        n = 2 ** k
        matrix = random_sparse(n, n, density=density, random_state=0).todense()
        x = np.random.random((n, 1))
        matrix_compressed = compress_matrix(matrix, r=r, epsilon=epsilon)
        start = perf_counter_ns()
        y = matrix_vector_mult(matrix_compressed, x)
        times.append(perf_counter_ns() - start)
        print(f"Time: {times[-1]}")

    return times


def test_matrix_matrix_mult(max_k, density=0.1, r=2, epsilon=0.001):
    times = []

    for k in range(1, max_k + 1):
        print(f"### matrix_matrix_mult 2^{k} x 2^{k} ###")
        n = 2 ** k
        matrix = random_sparse(n, n, density=density, random_state=0).todense()
        matrix_compressed1 = compress_matrix(matrix, r=r, epsilon=epsilon)
        matrix_compressed2 = compress_matrix(matrix, r=r, epsilon=epsilon)
        start = perf_counter_ns()
        y = matrix_matrix_mult(matrix_compressed1, matrix_compressed2, epsilon=epsilon)
        times.append(perf_counter_ns() - start)
        print(f"Time: {times[-1]}")

    return times


def test_matrix_matrix_add(max_k, density=0.1, r=2, epsilon=0.001):
    times = []

    for k in range(1, max_k + 1):
        print(f"### matrix_matrix_add 2^{k} x 2^{k} ###")
        n = 2 ** k
        matrix1 = random_sparse(n, n, density=density, random_state=0).todense()
        matrix2 = random_sparse(n, n, density=density, random_state=0).todense()
        matrix_compressed1 = compress_matrix(matrix1, r=r, epsilon=epsilon)
        matrix_compressed2 = compress_matrix(matrix2, r=r, epsilon=epsilon)
        start = perf_counter_ns()
        y = matrix_matrix_add(matrix_compressed1, matrix_compressed2, epsilon=epsilon)
        times.append(perf_counter_ns() - start)
        print(f"Time: {times[-1]}")

    return times


def func(x, a, n):
    return a * x ** n


def plot(max_k, times, title):
    x = np.arange(1, max_k + 1)

    popt, pcov = curve_fit(func, 2 ** x, times)
    a, n = popt
    x2 = np.linspace(0, max_k)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=130)
    ax.plot(x, times, "o", label=f"Czas wykonania")
    ax.plot(x2, func(2 ** x2, a, n), label=f"Dopasowana krzywa: {round(a, 3)} * N^{round(n, 3)}")
    ax.set_title(f"Macierze 2^k x 2^k, {title} - pomiar czasu i ocena złożoności przez dopasowanie krzywej do wykresu", y=1.03)
    ax.set_xlabel("k")
    ax.set_ylabel("Czas [ns]")
    ax.set_yscale('log')
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()


def plot_matrix_vector_mult():
    MAX_K = 9

    times_10 = test_matrix_vector_mult(MAX_K, density=0.1)
    times_20 = test_matrix_vector_mult(MAX_K, density=0.2)
    times_30 = test_matrix_vector_mult(MAX_K, density=0.3)
    times_40 = test_matrix_vector_mult(MAX_K, density=0.4)
    times_50 = test_matrix_vector_mult(MAX_K, density=0.5)

    plot(MAX_K, times_10, "10% wartości niezerowych")
    plot(MAX_K, times_20, "20% wartości niezerowych")
    plot(MAX_K, times_30, "30% wartości niezerowych")
    plot(MAX_K, times_40, "40% wartości niezerowych")
    plot(MAX_K, times_50, "50% wartości niezerowych")


def plot_matrix_matrix_add():
    MAX_K = 8

    times_10 = test_matrix_matrix_add(MAX_K, density=0.1)
    times_20 = test_matrix_matrix_add(MAX_K, density=0.2)
    times_30 = test_matrix_matrix_add(MAX_K, density=0.3)
    times_40 = test_matrix_matrix_add(MAX_K, density=0.4)
    times_50 = test_matrix_matrix_add(MAX_K, density=0.5)

    plot(MAX_K, times_10, "10% wartości niezerowych")
    plot(MAX_K, times_20, "20% wartości niezerowych")
    plot(MAX_K, times_30, "30% wartości niezerowych")
    plot(MAX_K, times_40, "40% wartości niezerowych")
    plot(MAX_K, times_50, "50% wartości niezerowych")


def plot_matrix_matrix_mult():
    MAX_K = 7

    times_10 = test_matrix_matrix_mult(MAX_K, density=0.1)
    times_20 = test_matrix_matrix_mult(MAX_K, density=0.2)
    times_30 = test_matrix_matrix_mult(MAX_K, density=0.3)
    times_40 = test_matrix_matrix_mult(MAX_K, density=0.4)
    times_50 = test_matrix_matrix_mult(MAX_K, density=0.5)

    plot(MAX_K, times_10, "10% wartości niezerowych")
    plot(MAX_K, times_20, "20% wartości niezerowych")
    plot(MAX_K, times_30, "30% wartości niezerowych")
    plot(MAX_K, times_40, "40% wartości niezerowych")
    plot(MAX_K, times_50, "50% wartości niezerowych")


if __name__ == "__main__":
    # plot_matrix_vector_mult()
    # plot_matrix_matrix_add()
    plot_matrix_matrix_mult()


