from system_model import SystemModel
from db_controller import Controller
import numpy as np
import matplotlib.pyplot as plt
import time

# Initialize system and controller
sm = SystemModel()
sm.z = np.vstack((np.random.uniform(low=-2.0, high=2.0, size=(2,1)), np.zeros((2, 1))))
y = sm.step(np.zeros((2, 1)))
ctrl = Controller()

# Simulation parameters
t_final = 30  # 30 seconds simulation
t = 0
dt = sm.dt

# Arrays to store data
t_points = []
y_points = []
u_points = []

# Main simulation loop
while t < t_final:
    # Calculate control input
    u = ctrl.calculate_acceleration(y)
    
    # Update system
    y = sm.step(u)
    
    # Store data
    t_points.append(t)
    y_points.append(y.flatten())
    u_points.append(u.flatten())
    
    # Update time
    t += dt

# Convert to numpy arrays for plotting
t_points = np.array(t_points)
y_points = np.array(y_points)
u_points = np.array(u_points)

# Create plots
plt.figure(figsize=(12, 8))

# Position plots
plt.subplot(2, 1, 1)
plt.plot(t_points, y_points[:, 0], label='y₁')
plt.plot(t_points, y_points[:, 1], label='y₂')
plt.grid(True)
plt.legend()
plt.ylabel('Position')
plt.title('System Response')

# Control input plots
plt.subplot(2, 1, 2)
plt.plot(t_points, u_points[:, 0], label='u₁')
plt.plot(t_points, u_points[:, 1], label='u₂')
plt.grid(True)
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Control Input')

plt.tight_layout()
plt.show()