# -*- coding: utf-8 -*-
"""
@author: Helge-André Langåker
"""
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import casadi as ca
from Models.gpmpc_model import Model
from Controllers.GPMPC.gp_class import GP
from Controllers.GPMPC.mpc_class import MPC

def ode(x, u, z, p=None):
    # Model Parameters (hardcoded)
    length = 0.5         # Length of the pendulum
    masspole = 0.1       # Mass of the pendulum
    cart_mass = 1.0      # Mass of the cart
    gravity = 9.8        # Gravitational acceleration
    total_mass = masspole + cart_mass
    polemass_length = masspole * length

    # Unpack the state vector
    x_pos = x[0]         # Cart position
    x_dot = x[1]         # Cart velocity
    theta = x[2]         # Pole angle
    theta_dot = x[3]     # Pole angular velocity

    # Unpack the control input
    force = u[0]         # Assuming u is a 1D array, with the applied force as the control input

    # Intermediate calculations
    costheta = ca.cos(theta)
    sintheta = ca.sin(theta)

    # Calculate accelerations
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # Return the derivatives of the state variables (dx/dt)
    dxdt = [x_dot, xacc, theta_dot, thetaacc]
    
    return ca.vertcat(*dxdt)



""" System parameters """
dt = 0.02
Nx = 4
Nu = 1
R = np.eye(Nx) * 1e-5 # Noise covariance

""" Limits in the training data """

ulb = [-2]
uub = [2]
xlb = [-2, -2, -0.9, -2]
xub = [2, 2, 0.9, 2]

N = 100          # Number of training data
N_test = 100    # Number of test data

""" Create Simulation Model """
model          = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=False)
X, Y           = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)



""" Create GP model and optimize hyper-parameters on training data """
gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
        uub=uub)



""" Initial state, input and set point  """
x_sp = np.array([0, 0, 0, 0]) # this is the reference value
x0 = np.array([
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.01, high=0.01),
            np.random.uniform(low=-0.01, high=0.01)
        ])

u0 = np.array([0.0])

""" Penalty matrices """
Q = np.diag([10, 1, 20, 1])   # State penalty
R = np.diag([0.1])       # Input penalty
S = np.diag([0.01])         # Input change penalty

""" Options to pass to the MPC solver """
solver_opts = {
        #    'ipopt.linear_solver' : 'ma27',    # Plugin solver from HSL
            'ipopt.max_cpu_time' : 30,
            'expand' : True,
}

""" Build MPC solver """
mpc = MPC(horizon=30*dt, gp=gp, model=model,
           gp_method='TA',
           ulb=ulb, uub=uub, xlb=xlb, xub=xub, Q=Q, R=R, S=S,
           terminal_constraint=None, costFunc='quad', feedback=True,
           solver_opts=solver_opts, discrete_method='gp',
           inequality_constraints=None
           )

""" Solve and plot the MPC solution, simulating 80 iterations """
x, u = mpc.solve(x0, u0=u0, sim_time=500*dt, x_sp=x_sp, debug=False, noise=True)

# mpc.plot(xnames=['Tank 1 [cm]', 'Tank 2 [cm]','Tank 3 [cm]','Tank 4 [cm]'],
#         unames=['Pump 1 [ml/s]', 'Pump 2 [ml/s]'])

y_ref = np.array([0, 0, 0, 0])

# mpc.plot(xnames=['1', '2','3','4'],
#         unames=['Pump 1 [ml/s]'])

mpc.plot(xnames=['Tank 1 [cm]', 'Tank 2 [cm]','Tank 3 [cm]','Tank 4 [cm]'],
        unames=['Pump 1 [ml/s]'], filename='inverted_pendulum', y_ref=y_ref)
