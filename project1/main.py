from system_model import SystemModel
from controller import Controller
import numpy as np
import time

sm = SystemModel()
sm.z = np.vstack((np.random.uniform(low=-2.0, high=2.0, size=(2,1)), np.zeros((2, 1)))) # initialize state of the system
y = sm.step(np.zeros((2, 1))) # get initial robot position

ctrl = Controller()

while True:
    u = ctrl.calculate_acceleration(y)
    
    y = sm.step(u)

    time.sleep(sm.dt)