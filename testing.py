import numpy as np
import scipy as sp
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
from pyquaternion import Quaternion

# сделаем расчет коэффициентов А и В в формулах из первого пункта через модуль Orbit библиотеки poliastro
def get_ab_p(orbit):
    r = orbit.r
    v = orbit.v
    r_cross_v = np.linalg.cross(r, v)
    e_y = r_cross_v / np.linalg.norm(r_cross_v)
    e_x = v / np.linalg.norm(v)
    # так как все это уже задано в некоторой интерциальной системе (в случае с poliastro -- в GCRS), 
    # то считаем за x первый базисный вектор => первые компоненты векторов и дают искомое скалярное произведение
    return e_x[0], e_y[0]

# Здесь важно отметить, что такой расчет работает и для эллиптической орбиты 
# (так как при выводе соотноешний в пункте 1 мы существенно опирались на свойство перпендикулярности скорость и радиус-вектора аппарата) 

# Окончательный расчет с помощью библиотеки
def get_phi_p(orbit):
    A, B = get_ab_p(orbit)
    phi1 = np.arctan2(-A , B)  # пластина исходной стороной вверх
    phi2 = np.arctan2(A , -B)  # пластина обратной стороной вверх
    return phi1, phi2

# Расчет орбитальной системы по положунию и скорости аппарата
def get_orbital_frame_matrix(r, v):
    r_cross_v = np.linalg.cross(r, v)
    e_y = r_cross_v / np.linalg.norm(r_cross_v)
    e_x = v / np.linalg.norm(v)
    return np.array([e_x, e_y, np.linalg.cross(e_x, e_y)]).T

def get_eigen_frame_matrix(e_x, e_y, e_z, phi):
    # собственаая система опирается существенно на орбитальную, так как ориентация аппарата жестко к ней привязана
    t_y = -e_x * np.sin(phi) + e_y * np.cos(phi)
    t_x = np.linalg.cross(t_y, e_z)
    return np.array([t_x, t_y, e_z]).T

# Расчет матрицы перехода из собстсвенной системы аппарата в орбитальную систему
def get_eigen_2_orbital_matrix(r, v, phi):
    # сложим базисные вектора в матрицы переходов
    # эти матрицы задают преобразование поворота векторов из инерциального базиса 
    # в соотвествующий (то есть применяется активная точка зрения)
    orbital_matrix = get_orbital_frame_matrix(r, v)
    eigen_matrix = get_eigen_frame_matrix(orbital_matrix[:, 0], orbital_matrix[:, 1], orbital_matrix[:, 2], phi)
    # расчитаем матрицу перехода из одного базиса в другой
    # эта матрица перехода будет задавать поворот векторов из орбитального базиса в собственный 
    # но кажется, что это тривиальная задача
    transfrom_matrix = orbital_matrix.T @ eigen_matrix
    return transfrom_matrix

# Все описанное здесь будет работать и для произвольной ориентации космического объекта,
# для нашей же конкретной задачи это можно оптимизировать: 
# orbital -> eigen задается просто матрицей поворота вокруг оси e_z, так что нет смысла расчитывать вектора, можно сразу записать преобразование, 
# причем можно сразу посчитать и кватернион перехода   


