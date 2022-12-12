import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from scipy.optimize import curve_fit
from scipy.sparse import random as random_sparse

from compress import compress_matrix, decompress


def test_compression(max_k, density=0.1, r=1, epsilon=0.000001):
    times = []

    for k in range(1, max_k + 1):
        print(f"### TEST COMPRESSION 2^{k} x 2^{k} ###")
        n = 2 ** k
        matrix = random_sparse(n, n, density=density, random_state=0).todense()
        start = perf_counter_ns()
        matrix_compressed = compress_matrix(matrix, r=r, epsilon=epsilon)
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
    ax.plot(x2, func(2 ** x2, a, n), label=f"Dopasowana krzywa: {round(a, 3)} * x^{round(n, 3)}")
    ax.set_title(f"Macierze 2^k x 2^k, {title} - pomiar czasu i ocena złożoności przez dopasowanie krzywej do wykresu", y=1.03)
    ax.set_xlabel("k")
    ax.set_ylabel("Czas [ns]")
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    MAX_K = 10

    times_10 = test_compression(MAX_K, density=0.1)
    times_20 = test_compression(MAX_K, density=0.2)
    times_30 = test_compression(MAX_K, density=0.3)
    times_40 = test_compression(MAX_K, density=0.4)
    times_50 = test_compression(MAX_K, density=0.5)

    plot(MAX_K, times_10, "10% wartości niezerowych")
    plot(MAX_K, times_20, "20% wartości niezerowych")
    plot(MAX_K, times_30, "30% wartości niezerowych")
    plot(MAX_K, times_40, "40% wartości niezerowych")
    plot(MAX_K, times_50, "50% wartości niezerowych")


