from method import *
import numpy as np
import matplotlib.pyplot as plt


def lotka_volterra(_, x):
    alpha = 1.1
    beta = 0.4
    gamma = 0.4
    delta = 0.1

    xdot = np.array([alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]])

    return xdot


table = read_butcher_tableau()

# SOLUTION
x0 = np.array([20, 5], dtype=float)
t, y = rk(0, 70, x0, 0.01, lotka_volterra, table)

plt.subplot(1, 2, 1)
plt.plot(t, y[0, :], "r", label="Preys")
plt.plot(t, y[1, :], "b", label="Predators")
plt.xlabel("Time (t)")
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y[0, :], y[1, :])
plt.xlabel("Preys")
plt.ylabel("Predators")
plt.grid()

plt.show()
