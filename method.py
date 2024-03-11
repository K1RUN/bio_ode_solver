from typing import Callable
import fractions as fr
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


def get_k_coefficients(t: float, y: float, h: float, f: Callable[[float, float], float], tableau: dict) \
        -> list[float]:
    a_ = tableau['a_']
    c_ = tableau['c_']
    k = []
    t_n = t
    y_n = y
    for i in range(len(a_)):
        for j in range(len(a_[0])):
            if not math.fabs(a_[i][j] - 0) < 1e-10:
                y_n += h * a_[i][j] * k[j - 1]
        k.append(f(t_n + c_[i] * h, y_n))
        y_n = y
    return k


def rk_one_step(t: float, y: float, h: float, f: Callable[[float, float], float], tableau: dict) -> tuple[float, float]:
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    b_ = tableau['b_']
    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h
    return t_n, y_n
