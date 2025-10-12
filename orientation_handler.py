import numpy as np
from pyquaternion import Quaternion


class SpacecraftOrientationHandler():
    """
    Класс, предоставляющий функцонал для расчета
    1) оптимального угла поворота КА с точки зрения освещенности пластины
    2) значение освещенности при заданной ориентации в заданной точке орбиты
    
    Функционал работает только для круговых орбит (такое предположение
    делалось при выводе формул для освещённости и оптимального угла)

    Для некруговой орбиты есть два момента:
    1) Приведенная в задаче орбитальная система не позволит так просто выразить ориентацию,
       поскольку нормаль к плоскость xy не будет в зенит/надир. Нужно выбирать другую орбитальную систему
    2) Формулы будут аналогичные, но чуть более сложные 
       (нужно пересчитать проекции с учетом новой орбитальной системы)
    
    """
    def __init__(self, raduis, inc, raan, mu):
        self.inc = inc
        self.raan = raan
        self.p = raduis  # курговая орбита
        self.e = 0       # круговая орбита
        self.perip = 0   # круговая орбита
        self.mu = mu
        self.n = np.sqrt(mu) / self.p**(3/2)  # среднее движение (n = u'_t)

    def get_ab(self, u):
        """
        Расчитывает коэффициенты A и B для функции оптимального 
        угла поворота пластины с точки зрения освещенности.
        """
        a = (-1) * ( np.sin(u) * np.cos(self.raan) + np.cos(u) * np.abs(np.cos(self.inc)) * np.sin(self.raan) )
        b = np.abs(np.sin(self.inc)) * np.sin(self.raan)
        return a, b

    def optimal_phi(self, u):
        """
        Расситывает оптимальное значение угла поворота пластины 
        Избегаем сингулярность при помощи использования функции atan2
        """
        A, B = self.get_ab(u)
        phi1 = -np.arctan2(A , B)  # пластина исходной стороной вверх
        phi2 = -np.arctan2(-A , -B)  # пластина обратной стороной вверх
        return phi1 if self.raan <= np.pi else np.pi + phi2
    
    def derivative_of_optimal_phi(self, u):
        """
        Расчет значения происзодной оптимального угла поворота пластины
        по аргументу широты. Это однозначно конвертируется в происзводную по времени через 
        известное значение среднего движения
        """
        A, B = self.get_ab(u)
        A_B_sq = (A**2 / B**2)
        A_pr_B = (-1) * (np.cos(self.raan) * np.cos(u) - np.sin(u) * np.sin(self.raan) * np.abs(np.cos(self.inc))) / B
        return -A_pr_B / (1 + A_B_sq)
    
    def illuminance(self, u, phi):
        """
        Рассчитывает освещенность пластины под заданным углом
        Принимаем во внимание, что пластина односторонняя, а значит, что для некотрых 
        углов освещенность может быть нулевая
        """
        A, B = self.get_ab(u)
        value = -A * np.sin(phi) + B * np.cos(phi)
        return np.maximum(np.zeros_like(value), value)
    
    def orientation_in_orbital_frame(self, phi):
        """
        Кватернион, задающий поворот из орбитальной системы в собственную
        """
        res = Quaternion(axis=np.array([0, 0, 1]), angle=phi) if phi <= np.pi  \
              else Quaternion(axis=np.array([0, 0, -1]), angle=-phi)
        
        return res



if __name__ == '__main__':
    pass
