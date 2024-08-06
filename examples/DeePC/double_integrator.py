from plot import plot_results
from controllers.DeePC.DeePC import npDeePC, DeePC
from examples.DeePC.double_int_ref import DoubleIntegrator

import numpy as np
import os
import casadi as ca
from models.gpmpc_model import Model

# What deepc needs: input signal data, 
# output singal data, 
# output constraints, 
# input constraints, 
# prediction horizon, 
# length of initial trajectory, 
# dimension of system, 
# output signal dimension, 
# input signal dimension


def ode(x, u, z, p):
    # Double integrator dynamics
    dxdt = [
        x[1],
        u[0]
    ]
    return ca.vertcat(*dxdt)

""" System parameters """
dt = 0.1
Nx = 2
Nu = 1
R = np.eye(Nx) * 1e-5 # Noise covariance



# first get the trajectories. 
T = 70  # length of trajectory
u_min = -1
u_max = 1
dt = 0.02
T_ini = 4
N = 20
p = 2
m = 1
total_simulation_time = 10
n_steps = int(total_simulation_time / dt)



def ode(x, u, z, p):
    # Double integrator dynamics
    dxdt = [
        x[1],
        u[0]
    ]
    return ca.vertcat(*dxdt)

""" System parameters """
dt = 0.1
Nx = 2
Nu = 1
R = np.eye(Nx) * 1e-5 # Noise covariance

# """ Limits in the training data """
# ulb = [-1.]
# uub = [1.]
# xlb = [-10., -10.]
# xub = [10., 10.]


# """ Create Simulation Model """
# model = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=False)
# U, Y = model.generate_training_data(T, uub, ulb, xub, xlb, noise=False, deepc=True)


double_integrator = DoubleIntegrator()


u_traj = np.random.uniform(u_min, u_max, (T, m)) 
initial_state = np.array([0, 0])
y_traj = double_integrator.generate_trajectory(initial_state, u_traj)

y_ref = double_integrator.generate_reference_trajectory(N)
print(y_ref)

print(u_traj)
print(y_traj)


# Define the constraints for one time step
y_upper_single = np.array([10, 10])  # Constraints for [position, velocity]
y_lower_single = np.array([-10, -10])
u_upper_single = np.array([u_max])  # Constraints for [acceleration]
u_lower_single = np.array([u_min])

# Extend the constraints over the prediction horizon using np.kron
y_upper = np.kron(np.ones(N), y_upper_single)
y_lower = np.kron(np.ones(N), y_lower_single)
u_upper = np.kron(np.ones(N), u_upper_single)
u_lower = np.kron(np.ones(N), u_lower_single)

# Combine into tuples for constraints
y_constraints = (y_lower, y_upper)
u_constraints = (u_lower, u_upper)

deepc = DeePC(u_traj, y_traj, y_constraints=y_constraints, u_constraints=u_constraints, N=N, Tini=T_ini)

Q = np.diag([50, 1])
R = np.diag([0.1])

deepc.setup(Q, R)

u_ini = np.array([0] * T_ini)
y_ini = np.array([0, 0] * T_ini)

u_ini = u_ini.reshape(T_ini * m, )
print("uini shape", u_ini.shape)
y_ini = y_ini.reshape(T_ini * p, )

final_position = 1.0  # Desired final position
final_velocity = 0.0  # Desired final velocity
y_ref = np.tile([final_position, final_velocity], N)
y_ref = y_ref.reshape(N * p, )
u_ref = np.zeros(N * m)

states = [y_ini[-p:]]
controls = []


for t in range(n_steps):
    new_u, _ = deepc.solve(y_ref, u_ref, u_ini, y_ini)

    print(new_u[:Nu])
    print(new_u.shape)
    print(u_ini[1:].shape)

    u_ini = np.vstack((u_ini[1:].reshape(-1, 1), new_u.reshape(-1,1))).flatten()
    y_ini = np.concatenate((y_ini.reshape(T_ini, p)[1:], double_integrator.step(y_ini[-p:], new_u).reshape(1, -1)),
                           axis=0).reshape(-1)

    new_y = y_ini[-p:]
    states.append(new_y)
    controls.append(new_u)



# Create a folder to store figures
cwd = os.getcwd()
figure_dir = os.path.join(cwd, 'figures')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

plot_results(np.array(states), np.array(controls), total_simulation_time, title="DeePC", with_limit_lines=False,
             with_video=False, figure_dir="figures", figure_name='Linear DeePC 1', linear=True)
