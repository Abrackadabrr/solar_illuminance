import numpy as np
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


class OrbitalFrameOrientationHandler():
    """
    Класс, предоставляющий функционал для расчета ориентации орбитальной системы
    на заданной орбите в кеплеровых элементах 
    (значения по умолчанию для двух последних аругментов соотвествуют круговой орбите)
    """
    def __init__(self, p, inc, raan, e=0, w=0):
        self.inc = inc
        self.raan = raan
        self.p = p
        self.e = e
        self.a = p / (1 - e**2)
        self.perip = w
        self.earth_radius = 6371

    def classic_orbital_frame_orientation(self, u):
        q_1 = Quaternion(axis=[0, 0, 1], angle=self.raan) if self.raan < np.pi else Quaternion(axis=[0, 0, -1], angle=-self.raan)
        q_2 = Quaternion(axis=[1, 0, 0], angle=self.inc)
        q_3 = Quaternion(axis=[0, 0, 1], angle=u) if u < np.pi else Quaternion(axis=[0, 0, -1], angle=-u)
        return q_1 * q_2 * q_3

    def orbital_frame_orientation(self, u):
        """
        Расчитывает ориентацию орбитальной системы относительно инерциальной
        """
        # а этот кватернион нужен для того, чтобы правильно оси расставить 
        # в задаче орбитальная система немного другая, нежели стандартная
        q_4 = Quaternion(axis=[1, 1, 1], angle=2*np.pi/3)

        return self.classic_orbital_frame_orientation(u) * q_4

    def is_inside_the_dark(self, u):
        """
        Определяет, находится ли аппарат в тени
        """
        def get_result(r):
            return (r[2]**2 + r[1]**2 < self.earth_radius**2) and (r[0] < 0)
        
        if type(u) == np.ndarray:
            return self.a * np.array([get_result( self.classic_orbital_frame_orientation(us).rotate(np.array([self.a, 0, 0])) ) \
                      for us in u])

        return get_result(self.classic_orbital_frame_orientation(u).rotate(np.array([self.a, 0, 0])))

    def is_outside_the_dark(self, u):
        """
        Определяет, находится ли аппарат вне тени
        """
        def get_result(r):
            return (r[2]**2 + r[1]**2 >= self.earth_radius**2) or (r[0] >= 0)
        
        if type(u) == np.ndarray:
            return np.array([get_result( self.classic_orbital_frame_orientation(us).rotate([self.a, 0, 0]) ) \
                      for us in u])

        return get_result(self.classic_orbital_frame_orientation(u).rotate([self.a, 0, 0]))


if __name__ == "__main__":
    pass
