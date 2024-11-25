import numpy as np
from scipy import signal

class Controller:
    def __init__(self):
        # State space matrices
        self.A = np.array([[0, 1], [-0.005, -1.05]])
        self.B = np.array([[0], [0.53]])
        
        # State feedback gains (computed from previous code)
        self.K = np.array([[3.09416374, 1.79245283]])
        
        # Low-pass filter state space matrices (for Item 4)
        self.Af = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [-1000, -300, -30]])
        self.Bf = np.array([[0],
                           [0],
                           [1000]])
        self.Cf = np.array([1, 0, 0])
        self.Df = np.array([0])
        
        # Initialize filter states
        self.xf1 = np.zeros((3, 1))
        self.xf2 = np.zeros((3, 1))
        
        # Store previous position measurements for velocity estimation
        self.prev_y = None
        self.dt = 0.001  # sampling time
        
        # For storing data for plotting
        self.y_history = []
        self.u_history = []
        self.t = 0
        
    def estimate_velocity(self, y):
        """Estimate velocity using finite differences"""
        if self.prev_y is None:
            self.prev_y = y
            return np.zeros((2, 1))
        
        velocity = (y - self.prev_y) / self.dt
        self.prev_y = y
        return velocity
    
    def filter_velocity(self, velocity, filter_index):
        """Apply low-pass filtering to velocity estimate"""
        if filter_index == 1:
            xf = self.xf1
        else:
            xf = self.xf2
            
        # Forward Euler integration for filter state
        xf = xf + (np.dot(self.Af, xf) + np.dot(self.Bf, velocity)) * self.dt
        
        # Compute filtered output
        velocity_filtered = np.dot(self.Cf, xf) + np.dot(self.Df, velocity)
        
        # Update filter state
        if filter_index == 1:
            self.xf1 = xf
        else:
            self.xf2 = xf
            
        return velocity_filtered
    
    def calculate_acceleration(self, y):
        """
        Calculate control input using state feedback
        y: robot position (2x1 numpy array)
        returns: control input (2x1 numpy array)
        """
        # Store data for plotting
        self.y_history.append(y.copy())
        self.t += self.dt
        
        # Reshape input if needed
        y = y.reshape(2, 1)
        
        # Estimate velocities
        velocity = self.estimate_velocity(y)
        
        # Filter velocities
        v1_filtered = self.filter_velocity(velocity[0], 1)
        v2_filtered = self.filter_velocity(velocity[1], 2)
        
        # Construct state vectors
        x1 = np.vstack((y[0].reshape(-1, 1), v1_filtered[0]))
        x2 = np.vstack((y[1], v2_filtered[0]))
        
        # Calculate control inputs using state feedback
        u1 = -np.dot(self.K, x1)
        u2 = -np.dot(self.K, x2)
        
        # Combine control inputs
        u = np.vstack((u1, u2))
        
        # Store control input for plotting
        self.u_history.append(u.copy())
        
        return u