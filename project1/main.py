from task3 import getK
from system_model import SystemModel
from controller import Controller
import matplotlib.pyplot as plt
import numpy as np
import time

sm = SystemModel()
# sm.z = np.vstack((np.random.uniform(low=-2.0, high=2.0, size=(2,1)), np.zeros((2, 1)))) # initialize state of the system
y = sm.step(np.zeros((2, 1))) # get initial robot position

K1 = getK()
K2 = K1

dt = 0.01
timePeriod = 30
t_arr = np.arange(0, timePeriod, dt)

u1_arr = []
u2_arr = []
y1_arr = []
y2_arr = []

ctrl = Controller(K1, K2, dt)

for t in t_arr:
    u = ctrl.calculate_acceleration(y)
    y = sm.step(u)

    u1_arr.append(u[0])
    u2_arr.append(u[1])
    y1_arr.append(y[0])
    y2_arr.append(y[1])

    time.sleep(sm.dt)

# Plot y1, y2, u1, u2 over time
plt.figure()
plt.plot(t_arr, y1_arr, label='y1')
plt.plot(t_arr, y2_arr, label='y2')
plt.xlabel('Time (s)')
plt.ylabel('Output')
plt.legend()
plt.title('System Outputs')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(t_arr, u1_arr, label='u1')
plt.plot(t_arr, u2_arr, label='u2')
plt.xlabel('Time (s)')
plt.ylabel('Control Input')
plt.legend()
plt.title('Control Inputs')
plt.grid(True)
plt.show()