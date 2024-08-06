# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import casadi as ca
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from controllers.DeePC2.deepc import DeePC
from controllers.DeePC2.utils import Data

from utils import System

# Define the loss function for DeePC
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]

    Q = np.eye(P)*10
    R = np.eye(M)*0.1

    Q = np.kron(np.eye(horizon), Q)
    R = np.kron(np.eye(horizon), R)
    # Sum_t ||y_t - 1||^2
    y_ref = np.ones((horizon, P))
    return 10* cp.norm(y-1,'fro')**2 + 0.1 * cp.norm(u, 'fro')**2
    # return cp.quad_form(y - y_ref, cp.psd_wrap(Q)) + cp.quad_form(u, cp.psd_wrap(R))


# Define the constraints for DeePC
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define input constraints
    
    return [u>=-2, u<=2, y<=5, y>=-5]

# DeePC paramters
s = 1                       # How many steps before we solve again the DeePC problem
T_INI = 4                   # Size of the initial set of data
T_list = [500, 500]          # Number of data points used to estimate the system
HORIZON = 10                # Horizon length
LAMBDA_G_REGULARIZER = 0    # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0  # y regularizer (see DeePC paper, eq. 8)
LAMBDA_U_REGULARIZER = 0   # u regularizer
EXPERIMENT_HORIZON = 100    # Total number of steps

# Plant
# In this example we consider the three-pulley 
# system analyzed in the original VRFT paper:
# 
# "Virtual reference feedback tuning: 
#      a direct method for the design of feedback controllers"
# -- Campi et al. 2003


dt = 0.05  # Example sampling time
A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
            [1]])
C = np.array([[1, 0]])
D = np.array([[0]])
I = np.eye(A.shape[0])
Ad = I + A*dt + (A @ A)*(dt**2)/2
Bd = (I*dt + (A*dt**2)/2) @ B

# Continuous-time transfer function of a double integrator
# num = [1]  # Numerator coefficients (1 for double integrator)
# den = [1, 0, 0]  # Denominator coefficients corresponding to 1/s^2

# Create the transfer function
sys = System(scipysig.StateSpace(A, B, C, D).to_discrete(dt))


fig, ax = plt.subplots(1,2)
plt.margins(x=0, y=0)


# Simulate for different values of T
for T in T_list:
    print(f'Simulating with {T} initial samples...')
    sys.reset()
    # Generate initial data and initialize DeePC
    data = sys.apply_input(u = np.random.normal(size=T).reshape((T, 1)), noise_std=0)
    deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

    # Create initial data
    data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
    sys.reset(data_ini = data_ini)

    deepc.build_problem(
        build_loss = loss_callback,
        build_constraints = constraints_callback,
        lambda_g = LAMBDA_G_REGULARIZER,
        lambda_y = LAMBDA_Y_REGULARIZER,
        lambda_u = LAMBDA_U_REGULARIZER)

    for idx in range(EXPERIMENT_HORIZON // s):
        # Solve DeePC
        print(data_ini.y)
        u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True)
        # Apply optimal control input
        print("u opt", u_optimal[:s,:])
        _ = sys.apply_input(u = u_optimal[:s, :], noise_std=1e-2)
        # print("y", _.y[:s,:])

        # Fetch last T_INI samples
        data_ini = sys.get_last_n_samples(T_INI)

    # Plot curve
    data = sys.get_all_samples()
    ax[0].plot(data.y[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')
    ax[1].plot(data.u[T_INI:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')


ax[0].set_ylim(0, 2)
ax[1].set_ylim(-4, 4)
ax[0].set_xlabel('t')
ax[0].set_ylabel('y')
ax[0].grid()
ax[1].set_ylabel('u')
ax[1].set_xlabel('t')
ax[1].grid()
ax[0].set_title('Closed loop - output signal $y_t$')
ax[1].set_title('Closed loop - control signal $u_t$')
ax[0].legend(fancybox=True, shadow=True)
plt.show()
