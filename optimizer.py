import numpy as np
from scipy.optimize import dual_annealing
from scipy.optimize import minimize

from orbital_handler import OrbitalFrameOrientationHandler
from orientation_handler import SpacecraftOrientationHandler


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
        a_s = np.logspace(3, 20, 17)
        previous_res = None
        for alpha in a_s:
            result = dual_annealing(minimized_function, bounds=bounds, args=(alpha, parametrization, derivative)) 
            if previous_res == None:
                previous_res = result
            else:
                if np.linalg.norm(result.x-previous_res.x) < self.tolerance:
                    return result
        return result
