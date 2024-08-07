from utils import System, Data
import scipy.signal as scipysig
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



# double integrator

# VARIABLES
T=70 # length of trajectory
N = 20
m=1 # input dimension
p=1 # output dimension
u_min = -1
u_max = 1
T_ini = 4
total_simulation_time = 10
dt = 0.05  # Example sampling time
n_steps = int(total_simulation_time // dt)

# VARIABLES

A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
            [1]])
C = np.array([[1, 0]])
D = np.array([[0]])




sys = System(scipysig.StateSpace(A, B, C, D).to_discrete(dt))

u_traj = np.random.uniform(u_min, u_max, (T, m))

# training data
training_data = sys.apply_input(u = u_traj, noise_std=0) # returns Data(u,y)




# generate the reference trajectory. VARIABLE shape Nxp
y_ref = np.ones((N, p)) # reference trajectory
u_ref = np.zeros((N, m)) # reference control input

# constraints VARIABLE
# Define the constraints for one time step
y_upper_single = np.array([10])  # Constraints for [position, velocity]
y_lower_single = np.array([-10])
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




deepc = DeePC(training_data.u.reshape(T*m,), training_data.y.reshape(T*p,), y_constraints=y_constraints, u_constraints=u_constraints, N=N, Tini=T_ini, p=p, m=m)




# VARIABLE

Q = np.diag([10])
R = np.diag([0.1])



deepc.setup(Q, R)




# u_ini, y_ini COULD BE VARIABLE
data_ini = Data(u = np.zeros((T_ini, m)), y = np.zeros((T_ini, p)))





y_ref = y_ref.reshape(N * p, )
u_ref = u_ref.reshape(N * m, )


sys.reset(data_ini = data_ini)




for t in range(n_steps):

    u_ini = sys.get_last_n_samples(T_ini).u.reshape(T_ini*m, )
    y_ini = sys.get_last_n_samples(T_ini).y.reshape(T_ini*p, )
    
    new_u, _ = deepc.solve(y_ref, u_ref, u_ini, y_ini)
    new_u = new_u.reshape(-1, m)

    sys.apply_input(new_u, noise_std=0)


states, controls = sys.get_all_samples().y, sys.get_all_samples().u


# Create a folder to store figures
cwd = os.getcwd()
figure_dir = os.path.join(cwd, 'figures')
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

plot_results(np.array(states), np.array(controls), total_simulation_time, title="DeePC", with_limit_lines=False,
             with_video=False, figure_dir="figures", figure_name='Linear DeePC 1', linear=True)




# # first get the trajectories. 
# T = 70  # length of trajectory
# u_min = -1
# u_max = 1
# dt = 0.02
# T_ini = 4
# N = 20
# p = 1
# m = 1
# total_simulation_time = 10
# n_steps = int(total_simulation_time / dt)



# double_integrator = DoubleIntegrator()





# u_traj = np.random.uniform(u_min, u_max, (T, m)) 
# initial_state = np.array([0, 0])
# y_traj = double_integrator.generate_trajectory(initial_state, u_traj)

# y_ref = double_integrator.generate_reference_trajectory(N)
# print(y_ref)

# print(u_traj)
# print(y_traj)


# # Define the constraints for one time step
# y_upper_single = np.array([10, 10])  # Constraints for [position, velocity]
# y_lower_single = np.array([-10, -10])
# u_upper_single = np.array([u_max])  # Constraints for [acceleration]
# u_lower_single = np.array([u_min])

# # Extend the constraints over the prediction horizon using np.kron
# y_upper = np.kron(np.ones(N), y_upper_single)
# y_lower = np.kron(np.ones(N), y_lower_single)
# u_upper = np.kron(np.ones(N), u_upper_single)
# u_lower = np.kron(np.ones(N), u_lower_single)

# # Combine into tuples for constraints
# y_constraints = (y_lower, y_upper)
# u_constraints = (u_lower, u_upper)

# deepc = DeePC(u_traj, y_traj, y_constraints=y_constraints, u_constraints=u_constraints, N=N, Tini=T_ini)

# Q = np.diag([50, 1])
# R = np.diag([0.1])

# deepc.setup(Q, R)

# u_ini = np.array([0] * T_ini)
# y_ini = np.array([0, 0] * T_ini)

# u_ini = u_ini.reshape(T_ini * m, )
# y_ini = y_ini.reshape(T_ini * p, )

# final_position = 1.0  # Desired final position
# final_velocity = 0.0  # Desired final velocity
# y_ref = np.tile([final_position, final_velocity], N)
# y_ref = y_ref.reshape(N * p, )
# u_ref = np.zeros(N * m)

# states = [y_ini[-p:]]
# controls = []


# for t in range(n_steps):
#     new_u, _ = deepc.solve(y_ref, u_ref, u_ini, y_ini)

#     u_ini = np.vstack((u_ini[1:].reshape(-1, 1), new_u.reshape(-1,1))).flatten()
#     y_ini = np.concatenate((y_ini.reshape(T_ini, p)[1:], double_integrator.step(y_ini[-p:], new_u).reshape(1, -1)),
#                            axis=0).reshape(-1)

#     new_y = y_ini[-p:]
#     states.append(new_y)
#     controls.append(new_u)



# # Create a folder to store figures
# cwd = os.getcwd()
# figure_dir = os.path.join(cwd, 'figures')
# if not os.path.exists(figure_dir):
#     os.makedirs(figure_dir)

# plot_results(np.array(states), np.array(controls), total_simulation_time, title="DeePC", with_limit_lines=False,
#              with_video=False, figure_dir="figures", figure_name='Linear DeePC 1', linear=True)
