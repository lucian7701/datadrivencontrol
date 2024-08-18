import numpy as np
from Models.deepc_model_l import System
from Controllers.DeePC.DeePCExecutor import DeePCExecutor

import scipy.signal as scipysig

# Define system parameters
T = 70  # Length of trajectory
N = 20  # Number of time steps
m = 1   # Input dimension
p = 1   # Output dimension
T_ini = 4 # Length of initial data
total_simulation_time = 10 
dt = 0.05  # Sampling time
n_steps = int(total_simulation_time // dt)

# Define system matrices
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define constraints
y_max = np.array([10])  # Constraints for [position, velocity]
y_min = np.array([-10])
u_max = np.array([2])  # Constraints for [acceleration]
u_min = np.array([-2])

y_ref = np.ones((N, p))*7


# Create the system
scipysystem = scipysig.StateSpace(A, B, C, D).to_discrete(dt)
sys = System(scipysystem)


# Define cost matrices
Q = np.diag([10])
R = np.diag([0.1])


u_traj = np.random.uniform(u_min, u_max, (T, m))
training_data = sys.apply_input(u = u_traj, noise_std=0) # returns Data(u,y)

# Create DeePC executor
executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=u_min, u_max=u_max,
                         y_min=y_min, y_max=y_max, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R, y_ref=y_ref, training_data=training_data)

# Run the executor
executor.run()

# Plot the results
executor.plot()
