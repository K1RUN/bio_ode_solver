import sys
import io

import numpy as np
from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau
from bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause

import matplotlib.pyplot as plt

y0 = np.array([20, 5], dtype=float)

step = 1
methods = ['rk_midpoint', 'rk2', 'rk4', 'rk5', 'dp8']
colors = ['green', 'blue', 'red', 'black', 'yellow']
points = {method: {} for method in methods}

while step >= 0.001:
    for method in methods:
        input_string = method
        original_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO(input_string)
            table = parse_butcher_tableau()
        finally:
            sys.stdin = original_stdin

        t_method, y_method = rk(0, 70, y0, step, lotka_volterra_gause, table)
        
        points[method][step] = {'t': t_method, 'y': y_method}
        
    step /= 2

fig, axs = plt.subplots(2, 3, figsize=(240, 15))

for i, method in enumerate(methods):
    for step, data in points[method].items():
        t_points = data['t']
        y_points = data['y']
        axs[i // 3][i % 3].plot(t_points, y_points[0], label=f'{method}, step={step}', color=colors[i])
        axs[i // 3][i % 3].plot(t_points, y_points[1], label=f'{method}, step={step}', color=colors[i])
        axs[i // 3][i % 3].set(xlabel="Time (t)", ylabel="Population (N)")
        axs[i // 3][i % 3].grid(True)
        axs[i // 3][i % 3].set_title(f'{method}')

plt.grid(True)
plt.show()
