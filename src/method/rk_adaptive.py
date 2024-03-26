import numpy as np
from typing import Callable

from bio_ode_solver.src.method.rk import get_k_coefficients


def rk_adaptive(t0: float, t_end: float, y0: np.ndarray[float], h_init: float,
                f: Callable[[..., ...], np.ndarray], tableau: dict,
                Atoli: float, Rtoli: float) -> tuple[np.ndarray, np.ndarray]:
    t_limit = int((t_end - t0) / h_init)
    t = np.zeros(t_limit)
    y = np.zeros((y0.size, t_limit))
    y[:, 0] = y0
    h = h_init
    for step in range(t_limit - 1):
        t[step + 1], y[:, step + 1], h = rk_one_step_adaptive(float(t[step]), y[:, step], h, f, tableau, Atoli, Rtoli)

    return t, y


def rk_one_step_adaptive(t: float, y: np.ndarray, h: float, f: Callable[[..., ...], np.ndarray],
                         tableau: dict, Atoli: float, Rtoli: float) -> tuple[float, np.ndarray, float]:
    k = get_k_coefficients(t, y, h, f, tableau)
    y_n = y
    b_ = tableau['b_']
    #вложенные методы рк
    err_abs = np.sum([(b_[i] - tableau['b_second'][i]) * k[i] for i in range(len(b_))])

    err_rel = np.sqrt(np.sum([(err_abs / Atoli) ** 2, (err_abs / Rtoli) ** 2]))

    if err_rel > 1.0:
        h_new = h * (1.0 / err_rel) ** (1.0 / (len(b_) + 1))
    else:

        h_new = h * (1.0 / err_rel) ** (1.0 / (len(b_) + 1))

    for i in range(len(b_)):
        y_n += h * b_[i] * k[i]
    t_n = t + h
    return t_n, y_n, h_new
