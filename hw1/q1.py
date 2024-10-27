import numpy as np
import math
import matplotlib.pyplot as plt

dt = 0.01
endT = 10
t = 0
baselength = 2
state = np.array([0, 0, 0, 0, 0])

xpoints = [state[0]]
ypoints = [state[1]]

while t < endT:
    u = np.array([0.1, math.cos(t)])

    v = state[3]
    theta = state[2]
    delta = state[4]

    model = np.add(
        np.array([v * math.cos(theta), v * math.sin(theta), v * math.tan(delta) / baselength, 0, 0]),
        np.dot(np.array([[0, 0], [0, 0], [0, 0], [1, 0], [0, 1]]), u)
    )

    state = np.add(state, np.dot(model, dt))
    xpoints.append(state[0])
    ypoints.append(state[1])

    t = t + dt

timestamps = np.arange(0, endT + dt + dt, dt)

# plt.plot(xpoints, ypoints)
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')

# plt.plot(timestamps, xpoints)
# plt.ylabel('X Coordinate')
# plt.xlabel('Time')

plt.plot(timestamps, ypoints)
plt.ylabel('Y Coordinate')
plt.xlabel('Time')

plt.title('Kinematic Bicycle Model Simulation')
plt.grid()
plt.show()