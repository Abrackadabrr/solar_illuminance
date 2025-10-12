import numpy as np
import matplotlib.pyplot as plt

from orientation_handler import SpacecraftOrientationHandler
from orbital_handler import OrbitalFrameOrientationHandler


plt.style.use("/home/evgen/Education/MasterDegree/thesis/my_papers/Utils_for_papers/graph_style.mplstyle")


# 1) Задаем парметры круговой орбиты
mu = 398600.4415                               
a = 6371 + 800                                 
Omega = np.pi / 400
i = (90) * np.pi / 180

# 2) Создаем два хендлера
spacefraft_orientation = SpacecraftOrientationHandler(a, i, Omega, mu)
orbital_frame_orientation = OrbitalFrameOrientationHandler(a, i, Omega)

# 3) Моделируем один виток
# Один викок вокруг Земли -- это изменение аргумента широты от 0 до 2\pi
# Поэтому в качестве временной переменной был взят именно аргумент широты (банальное удобство)

discr_n = 500
u_s = np.linspace(0, 2 * np.pi, discr_n)

# угол между нормалью к панели и направлением на Солнце
phis = spacefraft_orientation.optimal_phi(u_s)
# освещенность панели на заданной ориентации
illuminance = spacefraft_orientation.illuminance(u_s, phis)
 # ориентация относительно орбитатальных осей
q_o2sc = np.array([spacefraft_orientation.orientation_in_orbital_frame(phi).elements for phi in phis])
print(q_o2sc)
 # ориентация относительно инерциальных осей
q_i2sc = np.array([(orbital_frame_orientation.orbital_frame_orientation(u) * 
          spacefraft_orientation.orientation_in_orbital_frame(phi)).elements for (u, phi) in zip(u_s, phis)])

# 4) Визуализация результатов

fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 14))

ax[0][0].plot(u_s, phis)
ax[0][0].set_title("Оптимальный угол поворота \n относительно оси z орбит. системы")

ax[1][0].plot(u_s, illuminance)
ax[1][0].set_title("Освещённость с оптимальной ориентацией")

ax[0][1].plot(u_s, q_o2sc[:,0], label='w')
ax[0][1].plot(u_s, q_o2sc[:, 1], label='x')
ax[0][1].plot(u_s, q_o2sc[:, 2], label='y')
ax[0][1].plot(u_s, q_o2sc[:, 3], label='z')
ax[0][1].set_title("Компоненты кватерниона из орбитальной системы \n в собственную систему КА")

ax[1][1].plot(u_s, q_i2sc[:,0], label='w')
ax[1][1].plot(u_s, q_i2sc[:, 1], label='x')
ax[1][1].plot(u_s, q_i2sc[:, 2], label='y')
ax[1][1].plot(u_s, q_i2sc[:, 3], label='z')
ax[1][1].set_title("Компоненты кватерниона из инерциальной системы \n в собственную ситсему КА")

for axes in ax:
    for axis in axes:
        axis.grid(True)
        axis.set_xlabel("Аргумент широты КА")
        axis.legend()
        ticks = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6, 
         np.pi, 7*np.pi/6, 4*np.pi/3, 3*np.pi/2, 5*np.pi/3, 11*np.pi/6, 2*np.pi]
        tick_labels = ['0', 'π/6', 'π/3', 'π/2', '2π/3', '5π/6', 
               'π', '7π/6', '4π/3', '3π/2', '5π/3', '11π/6', '2π']
        axis.set_xticks(ticks, tick_labels)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.2) 
plt.show()
