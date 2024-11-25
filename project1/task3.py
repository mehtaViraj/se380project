import control as ct
import numpy as np
import matplotlib.pyplot as plt

def getK():
    A1 = np.array([[0, 1], [-0.005, -1.05]])
    B1 = np.array([[0], [1]])
    # C1 = np.array([[0.53, 0]])
    # D1 = np.array([[0]])

    # stateSpace = ct.ss(A1, B1, C1, D1)

    # A2 = A1
    # B2 = B1
    # C2 = C1
    # D2 = D1

    # We have the formula for OS. Plug in OS%=2 and use Symbolab to solve for zeta
    zeta = 0.7797

    # We have the formula for Ts. Plug in Ts=4 and use Symbolab to solve for wn
    wn = 1.28254

    poles = [
        (-zeta*wn) + ( 1j * wn * np.sqrt( 1-(zeta**2) )),
        (-zeta*wn) - ( 1j * wn * np.sqrt( 1-(zeta**2) ))
    ]

    K1 = ct.place(A1, B1, poles)
    # K2 = K1

    print('K = ', K1)
    return K1
