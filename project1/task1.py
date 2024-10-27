from system_model import SystemModel
import numpy as np
import matplotlib.pyplot as plt
import control

sm = SystemModel()

input_signal = np.ones((15000, 2))
t = np.arange(0, 15000)
t_seconds = t * 0.001
# t_seconds = np.linspace(0, 15, int(15/0.001))

output_signal = sm.sim(input_signal)

# Using the formulae from Bode plots tau using bode plots
tau = (1/10) * (10**(1/10) - 1)**(1/2)

print(tau)

s = control.TransferFunction.s
G = 1 / (1 + tau * s)**3
print(G)

_, filtered_output_1 = control.forced_response(G, T=t_seconds, U=output_signal[:, 0])
_, filtered_output_2 = control.forced_response(G, T=t_seconds, U=output_signal[:, 1])

plt.plot(t, input_signal[:, 0], label='input_1')
plt.plot(t, output_signal[:, 0], label='output_1')
plt.plot(t, filtered_output_1, label='filtered_output_1')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude_1 (dB)')
plt.legend()
plt.show()

plt.plot(t, input_signal[:, 1], label='input_2')
plt.plot(t, output_signal[:, 1], label='output_2')
plt.plot(t, filtered_output_2, label='filtered_output_2')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude_2 (dB)')
plt.legend()
plt.show()