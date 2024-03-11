import matplotlib.pyplot as plt
from method import *
import numpy as np


def f(x, y):
    return x**2


def y1(x):
    return x**3/3. + 2/3.


x0 = 1.
y0 = 1.
dx = 25
x_end = 500

x_rk = [x0]
y_rk = [y0]
y = y0
x = x0

table = read_butcher_tableau()
while x <= x_end:
    x, y = rk_one_step(x, y, dx, f, table)
    x_rk.append(x)
    y_rk.append(y)


def Euler(x, y, dx, dydx):
    return x + dx, y + dx * dydx(x, y)


x_eu = [x0]
y_eu = [y0]

y = y0
x = x0

while x <= x_end:
    x, y = Euler(x, y, dx, f)

    x_eu.append(x)
    y_eu.append(y)


plt.figure(figsize=(7,5))

plt.plot(np.linspace(1,500,500), y1(np.linspace(1,500,500)),
         label="Analytical solution",color="red", lw=2)

plt.plot(x_rk, y_rk, label="Numerical solution:\nRunge-Kutta", dashes=(3,2), color="blue",
        lw=3)
plt.plot(x_eu, y_eu, label="Numerical solution:\nEuler", dashes=(3,2), color="green",
        lw=3)

plt.legend(loc="best", fontsize=12)
plt.title(r"Solution to ODE: $\quad\frac{dy}{dx}=x^2$")
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.show()
