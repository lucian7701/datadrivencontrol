import casadi as ca
from Models.deepc_model_nl import System
import numpy as np
from Controllers.DeePC.DeePCExecutor import DeePCExecutor

# Define system parameters
T = 500  # Length of trajectory
N = 20  # Number of time steps
m = 2   # Input dimension
p = 4   # Output dimension
T_ini = 4 # Length of initial data
total_simulation_time = 102
dt = 3  # Sampling time
n_steps = int(total_simulation_time // dt)


def ode(t, x, u):
    # Constants
    g = 981
    a1 = 0.233
    a2 = 0.242
    a3 = 0.127
    a4 = 0.127
    A1 = 50.27
    A2 = 50.27
    A3 = 28.27
    A4 = 28.27
    gamma1 = 0.4
    gamma2 = 0.4

    # State derivatives
    dxdt = [
            (-a1 / A1) * np.sqrt(2 * g * x[0] + 1e-3) + (a3 / A1)
                * np.sqrt(2 * g * x[2] + 1e-3) + (gamma1 / A1) * u[0],
            (-a2 / A2) * np.sqrt(2 * g * x[1] + 1e-3) + a4 / A2
                * np.sqrt(2 * g * x[3]+ 1e-3) + (gamma2 / A2) * u[1],
            (-a3 / A3) * np.sqrt(2 * g * x[2] + 1e-3) + (1 - gamma2) / A3 * u[1],
                (-a4 / A4) * np.sqrt(2 * g * x[3] + 1e-3) + (1 - gamma1) / A4 * u[0]
    ]

    return dxdt

x0 = np.array([8., 10., 8., 19.])




""" System parameters """
dt = 3.0
Nx = 4
Nu = 2

sys = System(ode=ode, m=Nu, Nx=Nx, dt=dt, x0=x0)

""" Limits in the training data """
ulb = [0., 0.]
uub = [60., 60.]
xlb = [0, 0, 0, 0]
xub = [30., 30., 30., 30.]

Q = np.diag([20, 20, 10, 10])   # State penalty
R = np.diag([1e-3, 1e-3])       # Input penalty

executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=ulb, u_max=uub,
                         y_min=xlb, y_max=xub, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R)

executor.run()

executor.plot()




