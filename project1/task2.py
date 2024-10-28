from system_model import SystemModel
import numpy as np
import matplotlib.pyplot as plt
import control

# Initialize the system model
sm = SystemModel()

# Define the time parameters
T = 15  # total time in seconds
dt = sm.dt  # time step size
N = int(T / dt)  # number of time steps

# Define unit step inputs for u1 and u2 separately
u1_step = np.zeros((N, 2))
u1_step[:, 0] = 1  # unit step on u1

u2_step = np.zeros((N, 2))
u2_step[:, 1] = 1  # unit step on u2

"""
TASK 1
"""

# Compute the output responses to the unit step inputs
y1 = sm.sim(u1_step)
y2 = sm.sim(u2_step)

y11 = y1[:, 0]
y12 = y1[:, 1]
y21 = y2[:, 0]
y22 = y2[:, 1]

# Plotting the outputs in 4 subplots
time = np.linspace(0, T, N)

plt.figure(figsize=(12, 10))

# Plot y11
plt.subplot(2, 2, 1)
plt.plot(time, y11, label="y11 (output y1 to step u1)", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y11: Output y1 to unit step input on u1")
plt.legend()

# Plot y12
plt.subplot(2, 2, 2)
plt.plot(time, y12, label="y12 (output y2 to step u1)", color='g')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y12: Output y2 to unit step input on u1")
plt.legend()

# Plot y21
plt.subplot(2, 2, 3)
plt.plot(time, y21, label="y21 (output y1 to step u2)", color='r')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y21: Output y1 to unit step input on u2")
plt.legend()

# Plot y22
plt.subplot(2, 2, 4)
plt.plot(time, y22, label="y22 (output y2 to step u2)", color='m')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y22: Output y2 to unit step input on u2")
plt.legend()

plt.tight_layout()
plt.show()

"""
TASK 2
"""

# Low-pass filter setup
tau = (1 / 10) 

# Define the transfer function for the low-pass filter
s = control.TransferFunction.s
G = 1 / (1 + tau * s) ** 3  # Third-order low-pass filter

# Apply the low-pass filter to each output signal using forced_response
_, filtered_output_y11 = control.forced_response(G, T=time, U=y11.flatten())
_, filtered_output_y12 = control.forced_response(G, T=time, U=y12.flatten())
_, filtered_output_y21 = control.forced_response(G, T=time, U=y21.flatten())
_, filtered_output_y22 = control.forced_response(G, T=time, U=y22.flatten())

# Plot the filtered outputs
plt.figure(figsize=(12, 10))

# Filtered y11
plt.subplot(2, 2, 1)
plt.plot(time, y11, label="Original y11", color='lightblue', linestyle='--')
plt.plot(time, filtered_output_y11, label="Filtered y11", color='b')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y11: Filtered Output y1 to unit step input on u1")
plt.legend()

# Filtered y12
plt.subplot(2, 2, 2)
plt.plot(time, y12, label="Original y12", color='lightgreen', linestyle='--')
plt.plot(time, filtered_output_y12, label="Filtered y12", color='g')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y12: Filtered Output y2 to unit step input on u1")
plt.legend()

# Filtered y21
plt.subplot(2, 2, 3)
plt.plot(time, y21, label="Original y21", color='lightcoral', linestyle='--')
plt.plot(time, filtered_output_y21, label="Filtered y21", color='r')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y21: Filtered Output y1 to unit step input on u2")
plt.legend()

# Filtered y22
plt.subplot(2, 2, 4)
plt.plot(time, y22, label="Original y22", color='violet', linestyle='--')
plt.plot(time, filtered_output_y22, label="Filtered y22", color='m')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("y22: Filtered Output y2 to unit step input on u2")
plt.legend()

plt.tight_layout()
plt.show()

"""
TASK 3
"""

# First derivatives of outputs
dy11_dt = np.diff(filtered_output_y11) / dt
dy12_dt = np.diff(filtered_output_y12) / dt
dy21_dt = np.diff(filtered_output_y21) / dt
dy22_dt = np.diff(filtered_output_y22) / dt

# Second derivatives of outputs by differentiating the first derivatives
d2y11_dt2 = np.diff(dy11_dt) / dt
d2y12_dt2 = np.diff(dy12_dt) / dt
d2y21_dt2 = np.diff(dy21_dt) / dt
d2y22_dt2 = np.diff(dy22_dt) / dt

# Pad first derivatives to match original array lengths
dy11_dt = np.append(dy11_dt, dy11_dt[-1])
dy12_dt = np.append(dy12_dt, dy12_dt[-1])
dy21_dt = np.append(dy21_dt, dy21_dt[-1])
dy22_dt = np.append(dy22_dt, dy22_dt[-1])

# Pad second derivatives to match original array lengths
d2y11_dt2 = np.append(d2y11_dt2, [d2y11_dt2[-1], d2y11_dt2[-1]])
d2y12_dt2 = np.append(d2y12_dt2, [d2y12_dt2[-1], d2y12_dt2[-1]])
d2y21_dt2 = np.append(d2y21_dt2, [d2y21_dt2[-1], d2y21_dt2[-1]])
d2y22_dt2 = np.append(d2y22_dt2, [d2y22_dt2[-1], d2y22_dt2[-1]])

# Plotting the first derivatives of the outputs
plt.figure(figsize=(12, 10))

# Plot dy11/dt
plt.subplot(2, 2, 1)
plt.plot(time, dy11_dt, label="First derivative dy11/dt", color='b')
plt.xlabel("Time (s)")
plt.ylabel("dy11/dt")
plt.title("First Derivative of y1 to unit step input on u1")
plt.legend()

# Plot dy12/dt
plt.subplot(2, 2, 2)
plt.plot(time, dy12_dt, label="First derivative dy12/dt", color='g')
plt.xlabel("Time (s)")
plt.ylabel("dy12/dt")
plt.title("First Derivative of y2 to unit step input on u1")
plt.legend()

# Plot dy21/dt
plt.subplot(2, 2, 3)
plt.plot(time, dy21_dt, label="First derivative dy21/dt", color='r')
plt.xlabel("Time (s)")
plt.ylabel("dy21/dt")
plt.title("First Derivative of y1 to unit step input on u2")
plt.legend()

# Plot dy22/dt
plt.subplot(2, 2, 4)
plt.plot(time, dy22_dt, label="First derivative dy22/dt", color='m')
plt.xlabel("Time (s)")
plt.ylabel("dy22/dt")
plt.title("First Derivative of y2 to unit step input on u2")
plt.legend()

plt.tight_layout()
plt.show()

# Plotting the second derivatives of the outputs
plt.figure(figsize=(12, 10))

# Plot d2y11/dt^2
plt.subplot(2, 2, 1)
plt.plot(time, d2y11_dt2, label="Second derivative d2y11/dt^2", color='b')
plt.xlabel("Time (s)")
plt.ylabel("d2y11/dt^2")
plt.title("Second Derivative of y1 to unit step input on u1")
plt.legend()

# Plot d2y12/dt^2
plt.subplot(2, 2, 2)
plt.plot(time, d2y12_dt2, label="Second derivative d2y12/dt^2", color='g')
plt.xlabel("Time (s)")
plt.ylabel("d2y12/dt^2")
plt.title("Second Derivative of y2 to unit step input on u1")
plt.legend()

# Plot d2y21/dt^2
plt.subplot(2, 2, 3)
plt.plot(time, d2y21_dt2, label="Second derivative d2y21/dt^2", color='r')
plt.xlabel("Time (s)")
plt.ylabel("d2y21/dt^2")
plt.title("Second Derivative of y1 to unit step input on u2")
plt.legend()

# Plot d2y22/dt^2
plt.subplot(2, 2, 4)
plt.plot(time, d2y22_dt2, label="Second derivative d2y22/dt^2", color='m')
plt.xlabel("Time (s)")
plt.ylabel("d2y22/dt^2")
plt.title("Second Derivative of y2 to unit step input on u2")
plt.legend()

plt.tight_layout()
plt.show()


"""
TASK 5
"""

def compute_derivative(signal, dt):
    return np.gradient(signal, dt)

# TASK 5: Function to calculate transfer function parameters and create transfer function
def create_transfer_function_from_data(output, d_output, d2_output, input_signal, d_input):
    # Form regression matrix by stacking output derivatives and input terms
    A = np.column_stack([
        # d2_output,            # a2 * y''
        d_output,             # a1 * y'
        output,               # a0 * y
        d_input,              # b1 * u'
        input_signal          # b0 * u
    ])
    
    # Solve for parameters directly using numpy linear algebra
    _, _, _, theta = np.linalg.lstsq(A, d2_output, rcond=None)  # Solve for [a2, a1, a0, b1, b0]
    
    # Construct transfer function using the parameters
    num = [theta[2], theta[3]]  # Numerator coefficients [b1, b0]
    den = [1, theta[0], theta[1]]  # Denominator coefficients [1, a1 a0]
    
    return control.TransferFunction(num, den)  # Return the transfer function

# Compute derivatives for input signals
du1 = compute_derivative(u1_step[:, 0], dt)
du2 = compute_derivative(u2_step[:, 1], dt)

# # Calculate transfer functions for G11, G12, G21, G22
# G11 = create_transfer_function_from_data(filtered_output_y11, dy11_dt, d2y11_dt2, u1_step[:, 0], du1)
# G21 = create_transfer_function_from_data(filtered_output_y12, dy12_dt, d2y12_dt2, u1_step[:, 0], du1)
# G12 = create_transfer_function_from_data(filtered_output_y21, dy21_dt, d2y21_dt2, u2_step[:, 1], du2)
# G22 = create_transfer_function_from_data(filtered_output_y22, dy22_dt, d2y22_dt2, u2_step[:, 1], du2)

def makeTransferFunction(y, y1, y2, u, du):    
    # Use data after the trim_index to avoid initial transients
    system_output = y2
    
    # Stack input-output relationships into a matrix
    regression_matrix = np.column_stack([
        -y1, # -dy/dt
        -y,  # -y(t)
        du,  # du
        u    # Input signal u(t)
    ])
    
    # Solve for coefficients using least squares
    coeff = np.linalg.lstsq(regression_matrix, system_output, rcond=None)[0]
    pole1, pole0, gain1, gain0 = coeff
    
    # Construct transfer function using the coefficients
    numerator = [gain1, gain0]
    denominator = [1, pole1, pole0]
    
    return control.TransferFunction(numerator, denominator)

# Define step inputs for both u1 and u2 inputs
step_input_u1 = np.ones_like(time)  # Step input for G11 and G21
step_input_u2 = np.ones_like(time)  # Step input for G12 and G22

# Estimate transfer functions for all four pairs
G11 = makeTransferFunction(filtered_output_y11, dy11_dt, d2y11_dt2, step_input_u1, du1)
G12 = makeTransferFunction(filtered_output_y12, dy12_dt, d2y12_dt2, step_input_u2, du2)
G21 = makeTransferFunction(filtered_output_y21, dy21_dt, d2y21_dt2, step_input_u1, du1)
G22 = makeTransferFunction(filtered_output_y22, dy22_dt, d2y22_dt2, step_input_u2, du2)

# Print the transfer functions
print("Transfer Function G11(s):", G11)
print("Transfer Function G21(s):", G21)
print("Transfer Function G12(s):", G12)
print("Transfer Function G22(s):", G22)

"""
TASK 6
"""

# Simulate step response for a given transfer function
def simulate_step_response(transfer_function, time_vector):
    _, response = control.step_response(transfer_function, time_vector)
    return response

# Simulate the step responses
response_G1_identified = simulate_step_response(G11, time)

response_G2_identified = simulate_step_response(G12, time)

response_G21_identified = simulate_step_response(G21, time)

response_G22_identified = simulate_step_response(G22, time)

# Plotting the step responses
plt.figure(figsize=(12, 8))

# G11 Response Comparison
plt.subplot(2, 2, 1)
plt.plot(time, filtered_output_y11, label='True G11', linestyle='--', color='blue')
plt.plot(time, response_G1_identified, label='Identified G11', color='red')
plt.title('Step Response of G11')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# G12 Response Comparison
plt.subplot(2, 2, 2)
plt.plot(time, filtered_output_y12, label='True G12', linestyle='--', color='blue')
plt.plot(time, response_G2_identified, label='Identified G12', color='red')
plt.title('Step Response of G12')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# G21 Response Comparison
plt.subplot(2, 2, 3)
plt.plot(time, filtered_output_y21, label='True G21', linestyle='--', color='blue')
plt.plot(time, response_G21_identified, label='Identified G21', color='red')
plt.title('Step Response of G21')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# G22 Response Comparison
plt.subplot(2, 2, 4)
plt.plot(time, filtered_output_y22, label='True G22', linestyle='--', color='blue')
plt.plot(time, response_G22_identified, label='Identified G22', color='red')
plt.title('Step Response of G22')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()


"""
TASK 7
"""

# Define the sinusoidal input signal
def sinusoidal_input(t):
    return np.sin(t)  # Sinusoidal input: sin(t)

# Simulate system response for a given input signal and transfer function
def simulate_response(transfer_function, input_signal, time_vector):
    time, response = control.forced_response(transfer_function, T=time_vector, U=input_signal)
    return time, response

# Generate the sinusoidal input signal
sinusoidal_signal = sinusoidal_input(time)
sinusoidal_input_vector = np.column_stack([sinusoidal_signal, sinusoidal_signal])

# Simulate the responses to the sinusoidal input
real_sin_response = sm.sim(sinusoidal_input_vector)
real_sin_response_0 = real_sin_response[:, 0]
real_sin_response_1 = real_sin_response[:, 1]
_, response_G1_identified = simulate_response(G11, sinusoidal_signal, time)
_, response_G2_identified = simulate_response(G22, sinusoidal_signal, time)

# Plotting the responses to sinusoidal input
plt.figure(figsize=(12, 8))

# G11 Response Comparison
plt.subplot(2, 2, 1)
plt.plot(time, real_sin_response_0, label='True G11', linestyle='--', color='blue')
plt.plot(time, response_G1_identified + 8, label='Identified G11', color='red')
plt.title('Response of G11 to Sinusoidal Input')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# G12 Response Comparison
plt.subplot(2, 2, 2)
plt.plot(time, real_sin_response_1, label='True G12', linestyle='--', color='blue')
plt.plot(time, response_G2_identified + 8, label='Identified G12', color='red')
plt.title('Response of G12 to Sinusoidal Input')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend()
plt.grid()

# Show plots
plt.tight_layout()
plt.show()