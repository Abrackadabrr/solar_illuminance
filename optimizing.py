import numpy as np
from scipy.optimize import dual_annealing

from orbital_handler import OrbitalFrameOrientationHandler
from orientation_handler import SpacecraftOrientationHandler

import matplotlib.pyplot as plt


class OptimizeSpacecraftOrientation():
    def __init__(self, ill, constraint, tolerance=1e-6):
        self.illuminance = ill
        self.cnstr = constraint
        self.tolerance = tolerance

    def optimize(self, parametrization, derivative):
        def minimized_function(w, alpha, pr, der):
            u_s = np.linspace(0, 2 * np.pi, 1000)
            tau = u_s[1] - u_s[0]

            # интегрирование средними прямоугольниками
            values = self.illuminance(u_s + tau/2, pr(u_s + tau/2, w)) * tau
            integral = np.sum(values)
            parametrisation_der = der(u_s, w)

            #итоговое выражение с функцией штрафа
            return -integral + alpha * max(0, np.max(np.abs(parametrisation_der)) - self.cnstr)**2
        
        bounds = [(-100, 100), (-100, 100), (-2, 2)]
        a_s = np.logspace(0, 10, 10)
        previous_res = None
        for alpha in a_s:
            result = dual_annealing(minimized_function, bounds=bounds, args=(alpha, parametrization, derivative)) 
            if previous_res == None:
                previous_res = result
            else:
                if np.linalg.norm(result.x-previous_res.x) < self.tolerance:
                    return result
        return result

if __name__ == '__main__':

    # 1) Задаем парметры круговой орбиты
    mu = 398600.4415                               
    a = 6371 + 800                                 
    Omega = np.pi / 2
    i = (90 - 23.44) * np.pi / 180

    # 2) Создаем два хендлера
    spacefraft_orientation = SpacecraftOrientationHandler(a, i, Omega, mu)
    orbital_frame_orientation = OrbitalFrameOrientationHandler(a, i, Omega)

    # 3) Решаем задачу оптимизации
    def phi(u, w):
        return w[2] * np.arctan(w[0] * np.sin(u) + w[1] * np.cos(u))

    def phi_der(u, w):
        return w[2] * (w[0] * np.cos(u) - w[1] * np.sin(u)) / (1 + (w[0] * np.sin(u) + w[1] * np.cos(u))**2)

    angular_velocity_constraint = np.pi / 180
    phi_dot_constraint = np.sqrt( angular_velocity_constraint**2 / spacefraft_orientation.n**2 - 1)

    print("Full cnsrt:", angular_velocity_constraint)
    print("Mean motion:", spacefraft_orientation.n)
    print("Phi dot cnsrt:", phi_dot_constraint)

    optimizer = OptimizeSpacecraftOrientation(spacefraft_orientation.illuminance, phi_dot_constraint)
    result = optimizer.optimize(phi, phi_der)

    # 4) Визуализируем результат
    u_s = np.linspace(0, 2 * np.pi, 10000)
    phi_opt = spacefraft_orientation.optimal_phi(u_s)
    phi_found = phi(u_s, result.x)

    plt.style.use("/home/evgen/Education/MasterDegree/thesis/my_papers/Utils_for_papers/graph_style.mplstyle")
    fig, ax = plt.subplots(nrows=3, figsize=(12, 16))

    ax[0].plot(u_s, phi_opt, label='optimal')
    ax[0].plot(u_s, phi_found, label='found with cnstr')
    ax[0].legend()
    ax[0].grid(True)

    ill_opt_int = np.sum(spacefraft_orientation.illuminance(u_s, phi_opt) * (u_s[1] - u_s[0]))
    ill_found_int = np.sum(spacefraft_orientation.illuminance(u_s, phi_found) * (u_s[1] - u_s[0]))

    ax[1].plot(u_s, spacefraft_orientation.illuminance(u_s, phi_opt), label=f'optimal: integral = {np.round(ill_opt_int, 4)}')
    ax[1].plot(u_s, spacefraft_orientation.illuminance(u_s, phi_found), label=f'found with cnsrt: integral = {np.round(ill_found_int, 4)}')
    ax[1].legend()
    ax[1].grid(True)

    ax[2].plot(u_s, spacefraft_orientation.derivative_of_optimal_phi(u_s), label='optimal derivative')
    ax[2].plot(u_s, phi_der(u_s, result.x), label='found derivative')
    ax[2].plot(u_s, np.ones_like(u_s) * optimizer.cnstr, 'r--')
    ax[2].plot(u_s, np.ones_like(u_s) * -optimizer.cnstr, 'r--')
    ax[2].set_ylim([np.min(phi_der(u_s, result.x)) * 1.2, np.max(phi_der(u_s, result.x)) * 1.2])
    ax[2].grid(True)
    ax[2].legend()


    plt.show()
