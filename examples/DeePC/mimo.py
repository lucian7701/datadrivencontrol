import numpy as np
from Models.deepc_model import System
from Controllers.DeePC.DeePCExecutor import DeePCExecutor

import scipy.signal as scipysig

# Define system parameters
T = 80  # Length of trajectory
N = 20  # Number of time steps
m = 2   # Input dimension
p = 2   # Output dimension
T_ini = 4 # Length of initial data
total_simulation_time = 50 
dt = 0.1  # Sampling time
n_steps = int(total_simulation_time // dt)
















# Define the MIMO system
# Define the state-space matrices
A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[1, 0],
              [0, 1]])
C = np.array([[1, 0],
              [0, 1]])
D = np.array([[0, 0],
              [0, 0]])

# Create the state-space system
sys = System(scipysig.StateSpace(A, B, C, D, dt=dt))


# Define constraints
u_min = np.array([-10, -10])  # Minimum values for both inputs
u_max = np.array([10, 10])    # Maximum values for both inputs

# Define output constraints
y_min = np.array([-10, -10])  # Minimum values for both outputs
y_max = np.array([10, 10])    # Maximum values for both outputs

# Create the system
sys = System(scipysig.StateSpace(A, B, C, D).to_discrete(dt))

Q = np.diag([50, 50])  # Higher penalty on the first state than the second
R = np.diag([0.1, 0.1])  # Lower penalty on the second control input




# Create DeePC executor
executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=u_min, u_max=u_max,
                         y_min=y_min, y_max=y_max, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R)

# Run the executor
executor.run()

# Plot the results
executor.plot()
