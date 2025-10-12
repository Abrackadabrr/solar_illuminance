import numpy as np
import scipy as sp
from scipy.optimize import dual_annealing
import matplotlib.pyplot as plt

plt.style.use("/home/evgen/Education/MasterDegree/thesis/my_papers/Utils_for_papers/graph_style.mplstyle")

def get_ab(i, omega, u):
    a = (-1) * ( np.sin(u) * np.cos(omega) + np.cos(u) * np.abs(np.cos(i)) * np.sin(omega) )
    b = np.abs(np.sin(i)) * np.sin(omega)
    return a, b


def func(phi, A, B):
    return -A * np.sin(phi) + B * np.cos(phi)


def phi(u, w):
    return w[2] * np.arctan(w[0] * np.sin(u) + w[1] * np.cos(u))


def phi_der(u, w):
    return w[2] * (w[0] * np.cos(u) - w[1] * np.sin(u)) / (1 + (w[0] * np.sin(u) + w[1] * np.cos(u))**2)


def phi_optimal(i, omega, u):
    A, B = get_ab(i, omega, u)
    phi1 = np.arctan2(-A , B)  # пластина исходной стороной вверх
    phi2 = np.arctan2(A , -B)  # пластина обратной стороной вверх
    return phi1 if omega < np.pi else phi2


def phi_optimal_der(i, omega, u):
    A, B = get_ab(i, omega, u)
    A_B_sq = (A**2 / B**2)
    A_pr_B = (-1) * (np.cos(omega) * np.cos(u) - np.sin(u) * np.sin(omega) * np.abs(np.cos(i))) / B
    return -A_pr_B / (1 + A_B_sq)


def minimized_function(w, i, omega, alpha, constr):
    u = np.linspace(0, 2 * np.pi, 1000)
    a, b = get_ab(i, omega, u)
    tau = u[1] - u[0]
    #values = -(phi(u, w) + phi_optimal(i, omega, u))**2 * tau
    values = func(phi(u + tau/2, w), a, b) * tau
    integral = np.sum(values)
    phi_der_numerical = phi_der(u, w)
    return -integral + alpha * max(0, np.max(np.abs(phi_der_numerical)) - constr)**2


omega = 0.
i = (90 - 23.44) * np.pi / 180
alpa = 10
constr = 10

optimal_a_wo_consrt = np.cos(omega) / (np.abs(np.sin(i)) * np.sin(omega))
optimal_b_wo_consrt = np.abs(np.cos(i)) / np.abs(np.sin(i))


initial_for_method = [optimal_a_wo_consrt, optimal_b_wo_consrt]
result = dual_annealing(minimized_function,
                        bounds=[(-2 * np.abs(optimal_a_wo_consrt), 2 * np.abs(optimal_a_wo_consrt)), (-2 * np.abs(optimal_b_wo_consrt), 2 * np.abs(optimal_b_wo_consrt)), (-2, 2)] , 
                        args=(i, omega, alpa, constr * 0.99))

u = np.linspace(0, 2 * np.pi, 10000)
phi_opt = phi_optimal(i, omega, u)
phi_found = phi(u, result.x)

diff_a = np.abs(np.abs(optimal_a_wo_consrt) - np.abs(result.x[0])) / np.abs(optimal_a_wo_consrt)
diff_b = np.abs(np.abs(optimal_b_wo_consrt) - np.abs(result.x[1])) / np.abs(optimal_b_wo_consrt)
                
print("Diff in a:", np.round(diff_a, 4), "Diff in b: ", np.round(diff_b, 4))

fig, ax = plt.subplots(nrows=3, figsize=(12, 16))

ax[0].plot(u, phi_opt, label='optimal')
ax[0].plot(u, phi_found, label='found with cnstr')
ax[0].legend()
ax[0].grid(True)

a, b = get_ab(i, omega, u)
f_opt_int = np.sum(func(phi_opt, a, b) * (u[1] - u[0]))
f_found_int = np.sum(func(phi_found, a, b) * (u[1] - u[0]))
ax[1].plot(u, func(phi_opt, a, b), label=f'optimal: integral = {np.round(f_opt_int, 4)}')
ax[1].plot(u, func(phi_found, a, b), label=f'found with cnsrt: integral = {np.round(f_found_int, 4)}')
ax[1].legend()
ax[1].grid(True)

ax[2].plot(u, phi_optimal_der(i, omega, u), label='optimal derivative')
ax[2].plot(u, phi_der(u, result.x), label='found derivative')
ax[2].plot(u, np.ones_like(u) * constr, 'r--')
ax[2].plot(u, np.ones_like(u) * -constr, 'r--')
ax[2].set_ylim([np.min(phi_der(u, result.x)) * 1.2, np.max(phi_der(u, result.x)) * 1.2])
ax[2].grid(True)
ax[2].legend()
plt.show()
