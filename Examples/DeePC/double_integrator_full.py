import numpy as np
from Models.deepc_linear_system import LinearSystem
from Controllers.DeePC.DeePCExecutor import DeePCExecutor

import scipy.signal as scipysig

# Define system parameters
T = 69  # Length of trajectory
N = 30  # Prediction horizon
m = 1   # Input dimension
p = 2   # Output dimension
T_ini = 2 # Length of initial data
total_simulation_time = 20 
dt = 0.1  # Sampling time
n_steps = int(total_simulation_time // dt)

# Define system matrices
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0], [0, 1]])
D = np.array([[0], [0]])

# Define constraints
y_max = np.array([10, 10])  # Constraints for [position, velocity]
y_min = np.array([-10, -10])
u_max = np.array([2])  # Constraints for [acceleration]
u_min = np.array([-2])

y_ref = np.array([7, 0])

# Repeat x_ref over the entire control horizon
y_ref = np.tile(y_ref, (N, 1))



# Create the system
scipysystem = scipysig.StateSpace(A, B, C, D).to_discrete(dt)

x0 = np.array([0, 0])

sys = LinearSystem(scipysystem, x0, m)



# Define cost matrices
Q = np.diag([10, 1])
R = np.diag([0.1])


training_data = sys.generate_training_data(T=T, u_min=u_min, u_max=u_max, y_min=y_min, y_max=y_max)
print(training_data.u, training_data.y)
x1 = np.array([0, 0])
sys.reset(x0=x1)

# Create DeePC executor
executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=u_min, u_max=u_max,
                         y_min=y_min, y_max=y_max, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R, y_ref=y_ref, training_data=training_data)

# Run the executor
executor.run()


state_labels = ['Position (m)', 'Velocity (m/s)']
control_labels = ['Force (N)']
ref_labels = ['Position ref (m)', 'Velocity ref (m/s)']

# Plot the results
# executor.plot()

executor.run_eval(state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels)
