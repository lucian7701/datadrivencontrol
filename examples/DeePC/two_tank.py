import numpy as np
from Models.deepc_model import System
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


# model of two-tank example
A = np.array([
        [0.70469, 0.     ],
        [0.24664, 0.70469]])
B = np.array([[0.75937], [0.12515]])
C = np.array([[0., 1.]])
D = np.zeros((C.shape[0], B.shape[1]))

sys = System(scipysig.StateSpace(A, B, C, D, dt=0.05))


# Define cost matrices
Q = np.diag([10])
R = np.diag([0.1])

y_max = np.array([10])  # Constraints for [position, velocity]
y_min = np.array([-10])
u_max = np.array([2])  # Constraints for [acceleration]
u_min = np.array([-2])

y_ref = np.ones((N, p))*3


# Create DeePC executor
executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=u_min, u_max=u_max,
                         y_min=y_min, y_max=y_max, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R, y_ref=y_ref)

# Run the executor
executor.run()

# Plot the results
executor.plot()
