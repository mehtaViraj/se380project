import numpy as np
import control
import matplotlib.pyplot as plt

class RobotSystem:
    def __init__(self):
        # System parameters from G11(s) = G22(s) = 0.53s^2 + 1.05s + 0.005
        # State space representation for G11 and G22
        self.A1 = np.array([[0, 1],
                           [-0.005, -1.05]])
        self.B1 = np.array([[0],
                           [1]])
        self.C1 = np.array([[0.53, 0]])
        self.D1 = np.array([[0]])
        
        # Same for G22 since G11 = G22
        self.A2 = self.A1
        self.B2 = self.B1
        self.C2 = self.C1
        self.D2 = self.D1
        
        # Calculate feedback gains for desired performance
        self.K1 = self.design_controller(self.A1, self.B1)
        self.K2 = self.K1  # Same gains since systems are identical
        
    def design_controller(self, A, B):
        """
        Design state feedback controller for desired specifications:
        - Overshoot < 2%
        - Settling time < 4s
        """
        # For OS < 2% and Ts < 4s:
        # Using OS% = 100*exp(-zeta*pi/sqrt(1-zeta^2)) < 2%
        # and Ts ≈ 4/(zeta*wn) < 4s
        
        zeta = 0.7797032674120722  # Damping ratio for < 2% overshoot
        wn = 1.2825391938129473    # Natural frequency for Ts < 4s
        
        # Desired closed-loop poles
        p1 = -zeta*wn + 1j*wn*np.sqrt(1-zeta**2)
        p2 = -zeta*wn - 1j*wn*np.sqrt(1-zeta**2)
        desired_poles = [p1, p2]
        
        # Calculate gains using place command from control library
        sys = control.ss(A, B, np.eye(2), np.zeros((2, 1)))
        K = control.acker(sys.A, sys.B,  desired_poles)
        print("K: ", K)
        return K

class Controller:
    def __init__(self, K1, K2, dt):
        self.K1 = K1
        self.K2 = K2
        self.dt = dt
        
        # Initialize filter states
        self.xf1 = np.zeros((3, 1))
        self.xf2 = np.zeros((3, 1))
        
        # Low-pass filter parameters
        self.A_f = np.array([[0, 1, 0],
                            [0, 0, 1],
                            [-1000, -300, -30]])
        self.B_f = np.array([[0],
                            [0],
                            [1]])
        self.C_f = np.array([[1000, 0, 0]])
        self.D_f = np.array([[0]])
        
        # Create state-space filter system using control library
        self.filter_sys = control.ss(self.A_f, self.B_f, self.C_f, self.D_f)
        
        # Previous values for derivative approximation
        self.prev_y1 = 0
        self.prev_y2 = 0
        
    def calculate_acceleration(self, y1, y2, t, use_filter=True):
        """
        Calculate control inputs with option for filtered or unfiltered derivatives
        """
        # Approximate derivatives
        y1_dot = (y1 - self.prev_y1) / self.dt
        y2_dot = (y2 - self.prev_y2) / self.dt
        
        if use_filter:
            # Update filter states using matrix operations
            # For y1_dot filter
            self.xf1 = self.xf1 + (np.dot(self.A_f, self.xf1) + 
                                  self.B_f * y1_dot) * self.dt
            
            # For y2_dot filter
            self.xf2 = self.xf2 + (np.dot(self.A_f, self.xf2) + 
                                  self.B_f * y2_dot) * self.dt
            
            # Get filtered derivatives
            y1_dot_filtered = np.dot(self.C_f, self.xf1) + self.D_f * y1_dot
            y2_dot_filtered = np.dot(self.C_f, self.xf2) + self.D_f * y2_dot
            
            # Extract scalar values from the filtered derivatives
            y1_dot = y1_dot_filtered[0, 0]
            y2_dot = y2_dot_filtered[0, 0]
        
        # Construct state vectors
        x1 = np.array([y1, y1_dot])
        x2 = np.array([y2, y2_dot])
        
        # Calculate control inputs
        u1 = -np.dot(self.K1, x1)
        u2 = -np.dot(self.K2, x2)
        
        # Store current measurements
        self.prev_y1 = y1
        self.prev_y2 = y2
        
        return u1, u2
    
    def reset(self):
        """Reset controller states"""
        self.xf1 = np.zeros((3, 1))
        self.xf2 = np.zeros((3, 1))
        self.prev_y1 = 0
        self.prev_y2 = 0

def simulate_system(robot, controller, T, dt, initial_conditions, use_filter=True):
    """
    Simulate the complete system
    """
    # Time vector
    t = np.arange(0, T, dt)
    steps = len(t)
    
    # Initialize storage arrays
    y1_hist = np.zeros(steps)
    y2_hist = np.zeros(steps)
    u1_hist = np.zeros(steps)
    u2_hist = np.zeros(steps)
    
    # Initial conditions
    x1 = initial_conditions[:2].reshape(-1, 1)
    x2 = initial_conditions[2:].reshape(-1, 1)
    
    # Create state-space systems for simulation
    sys1 = control.ss(robot.A1, robot.B1, robot.C1, robot.D1)
    sys2 = control.ss(robot.A2, robot.B2, robot.C2, robot.D2)
    
    # Simulation loop
    for i in range(steps):
        # Get current outputs
        y1 = float(robot.C1 @ x1)
        y2 = float(robot.C2 @ x2)
        
        # Store outputs
        y1_hist[i] = y1
        y2_hist[i] = y2
        
        # Calculate control inputs
        u1, u2 = controller.calculate_acceleration(y1, y2, t[i], use_filter)
        
        # Store inputs
        u1_hist[i] = u1
        u2_hist[i] = u2
        
        # Update states using forward Euler
        x1_dot = robot.A1 @ x1 + robot.B1 * u1
        x2_dot = robot.A2 @ x2 + robot.B2 * u2
        
        x1 = x1 + x1_dot * dt
        x2 = x2 + x2_dot * dt
    
    return t, y1_hist, y2_hist, u1_hist, u2_hist

def plot_results(t, y1, y2, u1, u2, title_suffix=""):
    """
    Plot system responses and control inputs
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot outputs
    ax1.plot(t, y1, 'b-', label='y1')
    ax1.plot(t, y2, 'r--', label='y2')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Output')
    ax1.grid(True)
    ax1.legend()
    ax1.set_title(f'System Outputs {title_suffix}')
    
    # Plot control inputs
    ax2.plot(t, u1, 'b-', label='u1')
    ax2.plot(t, u2, 'r--', label='u2')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Input')
    ax2.grid(True)
    ax2.legend()
    ax2.set_title(f'Control Inputs {title_suffix}')
    
    plt.tight_layout()
    return fig

def main():
    # Initialize system and parameters
    dt = 0.01
    T = 30.0
    robot = RobotSystem()
    controller = Controller(robot.K1, robot.K2, dt)
    
    # Initial conditions [y1, y1_dot, y2, y2_dot]
    initial_conditions = np.array([1.0, 0.0, -1.0, 0.0])
    
    # Simulate with filtered derivatives
    t, y1, y2, u1, u2 = simulate_system(robot, controller, T, dt, 
                                      initial_conditions, use_filter=True)
    
    # Plot results
    fig = plot_results(t, y1, y2, u1, u2, "(with filtering)")
    plt.show()
    
    # Reset controller and simulate without filtering for comparison
    controller.reset()
    t, y1_uf, y2_uf, u1_uf, u2_uf = simulate_system(robot, controller, T, dt, 
                                                   initial_conditions, use_filter=False)
    
    # Plot unfiltered results
    fig = plot_results(t, y1_uf, y2_uf, u1_uf, u2_uf, "(without filtering)")
    plt.show()

if __name__ == "__main__":
    main()