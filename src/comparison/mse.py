import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from bio_ode_solver.src.method.rk import rk
from bio_ode_solver.src.method.rk_adaptive import rk_adaptive
from bio_ode_solver.src.utils.parse_tableau import parse_butcher_tableau
from bio_ode_solver.src.model.lotka_volterra_gause import lotka_volterra_gause

steps = [0.001 * 2 ** (n - 1) for n in range(1, 11)]
prefix = 'butcher_tables/'
methods = ['rk_midpoint', 'rk2', 'rk2_ralston', 'rk4', 'rk5', 'dp8']

mse_values = {method: [] for method in methods}

y0 = np.array([20, 5], dtype=float)
adap_std = parse_butcher_tableau(prefix + 'dp')

for step in steps:
    t_dp, y_dp = rk_adaptive(0, 70, y0, step, lotka_volterra_gause, adap_std, Atoli=1e-6, Rtoli=1e-6)
    for method in methods:
        table = parse_butcher_tableau(prefix + method)
        t_method, y_method = rk(0, 70, y0, step, lotka_volterra_gause, table)
        mse = mean_squared_error(y_dp.T, y_method.T)
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
