from Models.deepc_nonlinear_system import NonLinearSystem
import numpy as np
from Controllers.DeePC.DeePCExecutor import DeePCExecutor


# Define system parameters
T = 300  # Length of trajectory
N = 30  # Number of time steps
m = 2   # Input dimension
p = 4   # Output dimension
T_ini = 4 # Length of initial data
total_simulation_time = 240
dt = 3 # Sampling time
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

x0 = np.array([8, 10, 8, 19])


""" System parameters """
Nx = 4
Nu = 2

sys = NonLinearSystem(x0, ode=ode, dt=dt, m=2)

""" Limits in the training data """
ulb = np.array([10, 10])
uub = np.array([60., 60.])
xlb = np.array([7.5, 7.5, 3.5, 4.5])
xub = np.array([22., 22., 22., 22.])

Q = np.diag([20, 20, 10, 10])   # State penalty
R = np.diag([0.001, 0.001])   # Input penalty

x_ref = np.array([14.0, 14.0, 14.2, 21.3])

# Repeat x_ref over the entire control horizon
x_ref = np.tile(x_ref, (N, 1))


training_data = sys.generate_training_data(T=T, u_min=ulb, u_max=uub, y_min=xlb, y_max=xub)
sys.reset(x0=x0)

# get ini data first
data_ini = sys.apply_input(u = np.tile(np.array([45, 45]), (T_ini, 1)))
sys.reset(x0=x0)

executor = DeePCExecutor(T=T, N=N, m=m, p=p, u_min=ulb, u_max=uub,
                         y_min=xlb, y_max=xub, T_ini=T_ini,
                         total_simulation_time=total_simulation_time,
                         dt=dt, sys=sys, Q=Q, R=R, lam_g1=10, lam_g2=10, lam_y=1,
                         y_ref=x_ref, data_ini=data_ini,
                         training_data=training_data
                         )

executor.run()

executor.plot()

state_labels = ['Tank 1 (m)', 'Tank 2 (m)', 'Tank 3 (m)', 'Tank 4 (m)']
control_labels = ['Pump 1 (m^3/s)', 'Pump 2 (m^3/s)']
ref_labels = ['Tank 1 ref (m)', 'Tank 2 ref (m)', 'Tank 3 ref (m)', 'Tank 4 ref (m)']

executor.run_eval(state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels, plot=True)
