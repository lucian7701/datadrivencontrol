from Models.deepc_nonlinear_system import NonLinearSystem
import numpy as np
from Controllers.DeePC.DeePCExecutor import DeePCExecutor
import math

# Define system parameters
T = 300  # Length of trajectory
N = 30  # Number of time steps
m = 1   # Input dimension
p = 4   # Output dimension
T_ini= 4 # Length of initial data
total_simulation_time = 10
dt = 0.02 # Sampling time
n_steps = int(total_simulation_time // dt)

theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

def ode(t, x, u):

    length = 0.5
    masspole = 0.1
    gravity = 9.8
    tau = 0.02  # Time step for simulation
    total_mass = masspole + 1.0  # Assume cart mass = 1.0
    polemass_length = masspole * length

    # Unpack the state vector
    x_pos, x_dot, theta, theta_dot = x
    
    # Unpack the control input
    force = u[0]  # Assuming `u` is a 1D array and force is the only control input
    
    # Calculate intermediate variables
    costheta = math.cos(theta)
    sintheta = math.sin(theta)
    
    # Calculate accelerations
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    
    # Return the derivatives of the state variables
    dxdt = [x_dot, xacc, theta_dot, thetaacc]
    
    return dxdt

x0 = np.array([
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.01, high=0.01),
            np.random.uniform(low=-0.01, high=0.01)
        ])

# output_function = lambda x: np.array([x[0], x[2]])
sys = NonLinearSystem(x0, ode=ode, dt=dt, m=m)



""" Limits in the training data """
ulb = np.array([-2])
uub = np.array([2])
ylb = np.array([-2, -2, -0.9, -2])
yub = np.array([2, 2, 0.9, 2])

# ylb = np.array([-x_threshold, -theta_threshold_radians])
# yub = np.array([x_threshold, theta_threshold_radians])

Q = np.diag([10, 1, 20, 1])   # State penalty
R = np.diag([0.1])   # Input penalty

y_ref = np.array([0, 0, 0, 0])
# Repeat x_ref over the entire control horizon
y_ref = np.tile(y_ref, (N, 1))


# need to add an output function here.
training_data = sys.generate_training_data_cartpole(T=T, u_min=-2, u_max=2, y_min=None, y_max=None)
sys.reset(x0=x0)


# get ini data first
data_ini = sys.apply_input(u = np.tile(np.array([0]), (T_ini, 1)))
sys.reset(x0=x0)

executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=ulb, u_max=uub,
                         y_min=ylb, y_max=yub, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R, lam_g2=300, lam_g1=300, lam_y=100000,
                         y_ref=y_ref, data_ini=data_ini,
                         training_data=training_data
                         )

executor.run()

executor.plot()
