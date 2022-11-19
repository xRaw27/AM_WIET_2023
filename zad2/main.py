import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from scipy.optimize import curve_fit

from inverse import inverse
from LU import LU
from det import det


def test_inverse(max_k):
    counts = []
    times = []

    for k in range(1, max_k + 1):
        print(f"### TEST RECURSIVE INVERSE MATRIX 2^{k} x 2^{k} ###")
        n = 2 ** k
        A = np.random.rand(n, n)
        start = perf_counter_ns()
        A_inv, count = inverse(A)
        times.append(perf_counter_ns() - start)
        counts.append(count)

        if k < 3:
            print(f"A_inv = \n {A_inv}")
        print(f"Atomic operations count: {count}")
        print(f"Correct?: {np.allclose(A_inv, np.linalg.inv(A), atol=1e-05)}\n")

    return counts, times


def test_lu(max_k):
    counts = []
    times = []

    for k in range(1, max_k + 1):
        print(f"### TEST RECURSIVE LU DECOMPOSITION MATRIX 2^{k} x 2^{k} ###")
        n = 2 ** k
        A = np.random.rand(n, n)
        start = perf_counter_ns()
        L, U, count = LU(A)
        times.append(perf_counter_ns() - start)
        counts.append(count)

        if k < 3:
            print(f"L = \n {L}")
            print(f"U = \n {U}")
        print(f"Atomic operations count: {count}")
        print(f"Correct?: {np.allclose(A, L @ U)}\n")

    return counts, times


def test_det(max_k):
    counts = []
    times = []

    for k in range(1, max_k + 1):
        print(f"### TEST RECURSIVE DETERMINANT CALCULATION MATRIX 2^{k} x 2^{k} ###")
        n = 2 ** k
        A = np.random.rand(n, n)
        start = perf_counter_ns()
        det_A, count = det(A)
        times.append(perf_counter_ns() - start)
        counts.append(count)

        print(f"det(A): {det_A}")
        print(f"Atomic operations count: {count}")
        print(f"Correct determinant: {np.linalg.det(A)}\n")

    return counts, times


def func(x, a, n):
    return a * x ** n


def plot(max_k, counts, times, title):
    x = np.arange(1, max_k + 1)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=130)
    ax.plot(x, times, "o")
    ax.set_title(f"{title} - Pomiar czasu dla macierzy 2^k x 2^k", y=1.03)
    ax.set_xlabel("k")
    ax.set_ylabel("Czas [ns]")
    ax.set_xticks(x, x)
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(12, 7), dpi=130)
    ax.plot(x, counts, "o")
    ax.set_title(f"{title} - Pomiar liczby operacji zmienno-przecinkowych dla macierzy 2^k x 2^k", y=1.03)
    ax.set_xlabel("k")
    ax.set_ylabel("Liczba operacji zmienno-przecinkowych")
    ax.set_xticks(x, x)
    plt.grid()
    plt.show()

    popt, pcov = curve_fit(func, 2 ** x, counts)
    a, n = popt
    x2 = np.linspace(0, max_k)

    fig, ax = plt.subplots(figsize=(12, 7), dpi=130)
    ax.plot(x, counts, "o", label=f"{title} - Zmierzona liczba operacji zmienno-przecinkowych")
    ax.plot(x2, func(2 ** x2, a, n), label=f"{title} - Dopasowana krzywa: {round(a, 3)} * x^{round(n, 3)}")
    ax.set_title(f"{title} - Ocena złożoności przez dopasowanie krzywej do "
                 f"wykresu liczby operacji zmienno-przecinkowych", y=1.03)
    ax.set_xlabel("k")
    ax.set_ylabel("Liczba operacji zmienno-przecinkowych")
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    MAX_K = 8

    inv_count, inv_times = test_inverse(MAX_K)
    plot(MAX_K, inv_count, inv_times, "Rekurencyjne odwracanie macierzy")

    lu_count, lu_times = test_lu(MAX_K)
    plot(MAX_K, lu_count, lu_times, "Rekurencyjna dekompozycja LU")

    det_count, det_times = test_det(MAX_K)
    plot(MAX_K, det_count, det_times, "Rekurencyjne obliczanie wyznacznika")
