import control as ct
import matplotlib.pyplot as plt
import math
ct.use_matlab_defaults()




s = ct.TransferFunction.s

G = 1 / ((1+s)**3)

# Control parameters
omega_c = 4.1

# Lead compensator
alpha_lead = 0.09
mu_lead = 0.15
T_lead = 1.15 / (omega_c * math.sqrt(alpha_lead))
C_lead = mu_lead * (1 + T_lead * s) / (1 + alpha_lead * T_lead * s)

# Lag compensator
mu_lag = 6.6
alpha_lag = 0.09
T_lag = 6 / omega_c  
C_lag = mu_lag * (1 + T_lag * s) / (1 + alpha_lag * T_lag * s)





plt.figure()
ct.bode_plot(G, margins=True, omega_limits=[0.01, 100])
ct.bode_plot(C_lead * G, margins=True, omega_limits=[0.01, 100])
ct.bode_plot(C_lag * G, margins=True, omega_limits=[0.01, 100])
ct.bode_plot(C_lead * C_lag * G, margins=True, omega_limits=[0.01, 100])

plt.figure()
t, y = ct.step_response(G / (1 + G))
_, y_le = ct.step_response(C_lead * G / (1 + C_lead * G), t)
_, y_la = ct.step_response(C_lag * G / (1 + C_lag * G), t)
_, y_lela = ct.step_response(C_lead * C_lag * G / (1 + C_lead * C_lag * G), t)
plt.plot(t, y, label='C = 1')
plt.plot(t, y_le, label='w/ lead compensator')
plt.plot(t, y_la, label='w/ lag compensator')
plt.plot(t, y_lela, label='w/ lead-lag compensator')
plt.grid()
plt.legend()

plt.show()