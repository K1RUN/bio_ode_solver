from typing import Callable
import fractions as fr
import numpy as np
import math


def read_butcher_tableau() -> dict:
    filename = input()
    with open(filename) as file:
        info = file.read().splitlines()

    def is_valid_number(s: str):
        """CHECK IF STRING CONTAINS A VALID NUMBER: POSITIVE OR NEGATIVE RATIONAL (WITH '/' SYMBOL) OR INTEGER"""
        values = s.split('/')
        return (len(values) == 1 or len(values) == 2) and all(num.lstrip('-+').isdigit() for num in values)

    table = {}

    for i, line in enumerate(info):
        """LAST LINE IN INFO STANDS FOR b COEFFICIENTS, b COEFFICIENTS WILL BE ADDED SEPARATELY AFTER THE CYCLE"""
        """https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods#Implicit_Runge%E2%80%93Kutta_methods"""
        row = line.split()
        if all(is_valid_number(string) for string in row):
            row = [float(fr.Fraction(number)) for number in row]
            if i != len(info) - 1:
                table.setdefault('c_', []).append(row[0])
                table.setdefault('a_', []).append(row[1:])

            elif i == len(info) - 1:
                table.setdefault('b_', []).extend(row)

    return table


def get_k_coefficients(t: float, y: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray], tableau: dict) \
        -> list[np.ndarray]:
    a_ = tableau['a_']
    c_ = tableau['c_']
    k = []
    t_n = t

    # NEED TO COPY IT; BECAUSE NUMPY MODIFIES INITIAL ARRAYS AND SOLUTION DIVERGES
    y_n = np.copy(y)
    for i in range(len(a_)):
        for j in range(len(a_[0])):
            if not math.fabs(a_[i][j] - 0) < 1e-10:
                y_n += h * a_[i][j] * k[j - 1]
        k.append(f(t_n + c_[i] * h, y_n))
        y_n = np.copy(y)
    return k


def rk_one_step(t: float, y: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray], tableau: dict) \
        -> tuple[float, np.ndarray]:
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    b_ = tableau['b_']

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h

    return t_n, y_n


def rk(t0: float, t_end: float, y0: np.ndarray[float], h: float,
       f: Callable[[..., ...], np.ndarray], tableau: dict) -> tuple[np.ndarray, np.ndarray]:
    t_limit = int((t_end - t0) / h)
    t = np.zeros(t_limit)

    y = np.zeros((y0.size, t_limit))
    y[:, 0] = y0

    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1] = rk_one_step(float(t[step]), y[:, step], h, f, tableau)

    return t, y
