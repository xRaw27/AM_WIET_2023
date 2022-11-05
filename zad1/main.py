from strassen import strassen, strassen_with_count
from binet import binet, binet_with_count
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from scipy.optimize import curve_fit


def test_for_random_2_to_power_k(k, f, with_count=False):
    print(f'Test for random 2^k x 2^k matrix\nk = {k}')

    n = 2 ** k
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    print(f'A:\n{A}\n')
    print(f'B:\n{B}\n')

    if with_count:
        C, count = f(A, B)
        print(f'Atomic operations count: {count}\n')
        print(f'C:\n{C}')
        return count
    else:
        start = perf_counter_ns()
        C = f(A, B)
        print(f'C:\n{C}')
        return perf_counter_ns() - start


def func(x, a, n):
    return a * x ** n


def plot(max_k):
    x = np.arange(1, max_k + 1)
    # y1_binet = [test_for_random_2_to_power_k(k, binet) for k in x]
    # y1_strassen = [test_for_random_2_to_power_k(k, strassen) for k in x]
    y1_binet = [179600, 516800, 2811800, 29481000, 135214500, 1092205100, 9057287400, 73960381600, 606882734000]  # binet
    y1_strassen = [202600, 596100, 2970200, 18934700, 110653600, 821264900, 5586606700, 38296430500, 277760904300]  # strassen
    # y2_binet = [test_for_random_2_to_power_k(k, binet_with_count, with_count=True) for k in x]
    # y2_strassen = [test_for_random_2_to_power_k(k, strassen_with_count, with_count=True) for k in x]
    y2_binet = [24, 256, 2304, 19456, 159744, 1294336, 10420224, 83623936, 670040064]  # binet
    y2_strassen = [79, 841, 7039, 53881, 395599, 2842921, 20195359, 142547161, 1002548719]  # strassen

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(x, y1_binet, 'o', label="Algorytm Bineta")
    ax.plot(x, y1_strassen, 'o', label="Algorytm Strassena")
    ax.set_title("Pomiar czasu dla macierzy 2^k x 2^k")
    ax.set_xlabel("k")
    ax.set_ylabel("Czas [ns]")
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(x, y2_binet, 'o', label="Algorytm Bineta")
    ax.plot(x, y2_strassen, 'o', label="Algorytm Strassena")
    ax.set_title("Pomiar liczby operacji zmienno-przecinkowych dla macierzy 2^k x 2^k")
    ax.set_xlabel("k")
    ax.set_ylabel("Liczba operacji zmienno-przecinkowych")
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()

    popt_binet, pcov_binet = curve_fit(func, 2 ** x, y2_binet)
    a_binet, n_binet = popt_binet
    popt_strassen, pcov_strassen = curve_fit(func, 2 ** x, y2_strassen)
    a_strassen, n_strassen = popt_strassen
    x2 = np.linspace(0, max_k)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(x, y2_binet, 'o', label="Algorytm Bineta - Zmierzona liczba operacji zmienno-przecinkowych")
    ax.plot(x, y2_strassen, 'o', label="Algorytm Strassena - Zmierzona liczba operacji zmienno-przecinkowych")
    ax.plot(
        x2, func(2 ** x2, a_binet, n_binet),
        label=f'Algorytm Bineta - Dopasowana krzywa: {round(a_binet, 3)} * x^{round(n_binet, 3)}'
    )
    ax.plot(
        x2, func(2 ** x2, a_strassen, n_strassen),
        label=f'Algorytm Strassena - Dopasowana krzywa: {round(a_strassen, 3)} * x^{round(n_strassen, 3)}'
    )
    ax.set_title("Ocena złożoności poprzez dopasowanie krzywej do wykresu liczby operacji zmienno-przecinkowych")
    ax.set_xlabel("k")
    ax.set_ylabel("Liczba operacji zmienno-przecinkowych")
    ax.set_xticks(x, x)
    ax.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    plot(9)

