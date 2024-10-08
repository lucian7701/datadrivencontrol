from __future__ import division
from __future__ import print_function

import numpy as np
import casadi as ca
from Models.gpmpc_model import Model
from Controllers.GPMPC.gp_class import GP
from Controllers.GPMPC.mpc_class import MPC

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

""" Limits in the training data """
ulb = [-2.]
uub = [2.]
xlb = [-10., -10.]
xub = [10., 10.]

N = 40       # Number of training data
# N_test = 100    # Number of test data

""" Create Simulation Model """
model = Model(Nx=Nx, Nu=Nu, ode=ode, dt=dt, R=R, clip_negative=False)
X, Y = model.generate_training_data(N, uub, ulb, xub, xlb, noise=True)
# X_test, Y_test = model.generate_training_data(N_test, uub, ulb, xub, xlb, noise=True)

""" Create GP model and optimize hyper-parameters on training data """
gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb, uub=uub)
# gp.save_model('models/gp_double_integrator')
# gp.validate(X_test, Y_test)
# gp.print_hyper_parameters()

""" Limits in the MPC problem """
ulb = [-2.]
uub = [2.]
xlb = [-10., -10.]
xub = [10., 10.]

""" Initial state, input and set point  """
x_sp = np.array([7.0, 0]) # this is the reference value
x0 = np.array([0.0, 0.0])
u0 = np.array([0.0])

""" Penalty matrices """
Q = np.diag([10, 1])   # State penalty
R = np.diag([0.1])   # Input penalty
S = np.diag([0.01])   # Input change penalty

""" Options to pass to the MPC solver """
solver_opts = {
    'ipopt.max_cpu_time': 30,
    'expand': True,
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
x, u = mpc.solve(x0, u0=u0, sim_time=200*dt, x_sp=x_sp, debug=False, noise=True)

y_ref = np.array([7.0, 0])

mpc.plot(xnames=['Position', 'Velocity'],
         unames=['Force'], filename='double_integrator', y_ref=y_ref)

# mpc.plot(xnames=['Position', 'Velocity'],
#          unames=['Force'])
