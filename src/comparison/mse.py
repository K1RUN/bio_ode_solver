import sys
import io

import numpy as np
from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau 
from bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


input_string = "dp8"
original_stdin = sys.stdin
try:
    sys.stdin = io.StringIO(input_string)
    table = parse_butcher_tableau()
finally:
    sys.stdin = original_stdin

y0 = np.array([20, 5], dtype=float)

steps = [0.1, 0.01, 0.001]
methods = ['rk_midpoint', 'rk2', 'rk4', 'rk5']

mse_values = {method: [] for method in methods}

for step in steps:
    t_dp8, y_dp8 = rk(0, 70, y0, step, lotka_volterra_gause, table)
    for method in methods:
        input_string = method
        original_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO(input_string)
            table = parse_butcher_tableau()
        finally:
            sys.stdin = original_stdin

        t_method, y_method = rk(0, 70, y0, step, lotka_volterra_gause, table)
        mse = mean_squared_error(y_dp8.T, y_method.T)
        # print(f"MSE for {method}: {mse} with step:{step}")
        mse_values[method].append(mse)

plt.figure(figsize=(10, 6))
for method in methods:
    plt.plot(steps, mse_values[method], marker='o', label=method)

plt.title('Comparison of Numerical Methods')
plt.xlabel('Step Size')
plt.ylabel('Mean Squared Error')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
