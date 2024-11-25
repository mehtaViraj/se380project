from controller import Controller
import matplotlib.pyplot as plt
import numpy as np
from task3 import CustomSystemModel

model = CustomSystemModel()
K1 = model.K
K2 = K1

dt = 0.01
timePeriod = 30
t_arr = np.arange(0, timePeriod, dt)

u1_arr = []
u2_arr = []
y1_arr = []
y2_arr = []

x1 = np.array([1.0, 0.0]) # initial conditions
x2 = np.array([-1.0, 0.0]) # initial conditions

print("initial x1: ", x1)
print("initial x2: ", x2)

ctrl = Controller(K1, K2, dt)

for t in t_arr:
    # Translate state to position
    y = np.array([
        np.dot(model.C, x1)[0],
        np.dot(model.C, x2)[0]
    ])

    # Evaluate next control input
    u = ctrl.calculate_acceleration(y)
    # u = ctrl.calculate_acceleration(y[0], y[1], t)

    # print("Ax:      ", np.dot(model.A, x1))
    # print("Bu:      ", np.dot(model.B, u[0]))
    # print("Ax + Bu: ", np.dot(model.A, x1) + np.dot(model.B, u[0]))
    # print("x1:      ", x1)
    # Update state
    dx1 = np.dot(model.A, x1) + np.dot(model.B, u[0])
    dx2 = np.dot(model.A, x2) + np.dot(model.B, u[1])
    x1 = x1 + (dx1 * dt)
    x2 = x2 + (dx2 * dt)
    # print("new x1:  ", x1)
    # print("----------")

    u1_arr.append(u[0])
    u2_arr.append(u[1])
    y1_arr.append(y[0])
    y2_arr.append(y[1])

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