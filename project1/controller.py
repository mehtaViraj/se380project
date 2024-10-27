import numpy as np

class Controller:
    def __init__(self):
        pass

    def calculate_acceleration(self, y):
        # inputs
        #   y: output of the SystemModel (2x1 numpy array)
        # outputs
        #   control input for the SystemModel (2x1 numpy array)

        # implement your controller here
        # ...
        return np.zeros((2, 1)) # dummy controller