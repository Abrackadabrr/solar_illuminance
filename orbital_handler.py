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
        self.perip = w

    def orbital_frame_orientation(self, u):
        """
        Расчитывает ориентацию орбитальной системы относительно инерциальной
        """
        q_1 = Quaternion(axis=[0, 0, 1], angle=self.raan) if self.raan < np.pi else Quaternion(axis=[0, 0, -1], angle=-self.raan)
        q_2 = Quaternion(axis=[1, 0, 0], angle=self.inc)
        q_3 = Quaternion(axis=[0, 0, 1], angle=u) if u < np.pi else Quaternion(axis=[0, 0, -1], angle=-u)
        # а этот кватернион нужен для того, чтобы правильно оси расставить 
        # в задаче орбитальная система немного другая, нежели стандартная
        q_4 = Quaternion(axis=[1, 1, 1], angle=2*np.pi/3)

        return q_1 * q_2 * q_3 * q_4


if __name__ == "__main__":
    pass
