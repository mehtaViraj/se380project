import numpy as np

class Controller:
    def __init__(self, K1, K2, dt):
        self.K1 = K1
        self.K2 = K2
        self.dt = dt

        self.filter_x1 = np.array([[0], [0], [0]])
        self.filter_x2 = np.array([[0], [0], [0]])

        # ABCD matrices for low-pass filter
        self.A_f = np.array([[0, 1, 0], [0, 0, 1], [-1000, -300, -30]])
        self.B_f = np.array([[0],
                            [0],
                            [1]])
        self.C_f = np.array([[1000, 0, 0]])
        self.D_f = np.array([[0]])

        # # Create state-space filter system using control library
        # self.filter_sys = control.ss(self.A_f, self.B_f, self.C_f, self.D_f)

        # Previous values for derivative approximation
        self.prev_y1 = 0
        self.prev_y2 = 0
        

    def calculate_acceleration(self, y, filter=True):
        y1 = y[0]
        y2 = y[1]
        # Approximate y_dot
        dy1 = (y1 - self.prev_y1) / self.dt
        dy2 = (y2 - self.prev_y2) / self.dt

        if filter:
            # newX = oldX + deltaX
            # deltaX = xdot * dt
            # xdot = Ax + Bu
            # u = y_dot (since we are filtering y_dot)
            self.filter_x1 = self.filter_x1 + (np.dot(self.A_f, self.filter_x1) + self.B_f * dy1) * self.dt
            self.filter_x2 = self.filter_x2 + (np.dot(self.A_f, self.filter_x2) + self.B_f * dy2) * self.dt

            # y_dot = Cx + Du = Cx (D = 0)
            dy1 = np.dot(self.C_f, self.filter_x1)[0, 0]
            dy2 = np.dot(self.C_f, self.filter_x2)[0, 0]



        # Update previous values for next iteration
        self.prev_y1 = y1
        self.prev_y2 = y2

        # x = [y, ydot]
        x1 = np.array([y1, dy1])
        x2 = np.array([y2, dy2])
        
        # u = -Kx
        u1 = -np.dot(self.K1, x1)
        u2 = -np.dot(self.K2, x2)

        return u1, u2