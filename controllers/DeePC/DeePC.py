""" https://github.com/michael-cummins/DeePC-HUNT/blob/main/deepc_hunt/controllers.py """

import numpy as np
# from custom_gym_envs.cartpole import CustomCartPoleEnv
import cvxpy as cp
# from gymnasium.wrappers import RecordVideo
import os
# from plot import plot_results
# from cartpole.cartpole_dynamics import LinearizedCartPole
from typing import Tuple
# from linear_dynamics.double_integrator import DoubleIntegrator


# Okay so what i want to do is change this to use casadi instead


def block_hankel(w: np.ndarray, L: int, d: int) -> np.ndarray:
    """
    Builds block Hankel matrix for column vector w of order L
    args:
        w = column vector
        p = dimension of each block in w
        L = order of hankel matrix
    """
    T = int(len(w) / d)
    if L > T:
        raise ValueError('L must be smaller than T')
    H = np.zeros((L * d, T - L + 1))
    for i in range(0, T - L + 1):
        H[:, i] = w[d * i:d * (L + i)]
    return H

# should deepc be able to just take the input signal data and output signal data,
# and infer the output signal dimension, input signal dimension and dimension of the system from that. 


class DeePC:

    """
    Vanilla regularized DeePC module
    """

    def __init__(self, ud: np.ndarray, yd: np.ndarray, y_constraints: Tuple[np.ndarray], u_constraints: Tuple[np.ndarray], 
                 N: int, Tini: int) -> None:
       
        """
        Initialise variables
        args:
            ud = Inpiut signal data
            yd = output signal data
            N = predicition horizon
            n = dimesnion of system
            p = output signla dimension
            m = input signal dimension
        """

        self.T = ud.shape[0]
        self.Tini = Tini
        # self.n = n 
        self.N = N
        # self.p = p
        # self.m = m
        self.p = yd.shape[1]  # Output signal dimension
        self.m = ud.shape[1]  # Input signal dimension
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]

        # Check for full row rank
        # H = block_hankel(w=ud.reshape((m*self.T,)), L=Tini+N+n, d=m)
        H = block_hankel(w=ud.reshape((self.m*self.T,)), L=Tini+N, d=self.m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')
        
        # Construct data matrices
        U = block_hankel(w=ud.reshape((self.m*self.T,)), L=Tini+N, d=self.m)
        Y = block_hankel(w=yd.reshape((self.p*self.T,)), L=Tini+N, d=self.p)
        self.Up = U[0:self.m*Tini,:]
        self.Yp = Y[0:self.p*Tini,:]
        self.Uf = U[Tini*self.m:,:]
        self.Yf = Y[Tini*self.p:,:]

        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.g = cp.Variable(self.T-self.Tini-self.N+1)
        self.y = cp.Variable(self.N*self.p)
        self.sig_y = cp.Variable(self.Tini*self.p)

        self.y_ref = cp.Parameter((self.N*self.p,))
        self.u_ref = cp.Parameter((self.N*self.m,))
        self.u_ini = cp.Parameter(self.Tini*self.m)
        self.y_ini = cp.Parameter(self.Tini*self.p)

        # Regularization Variables
        PI = np.vstack([self.Up, self.Yp, self.Uf])
        PI = np.linalg.pinv(PI)@PI
        I = np.eye(PI.shape[0])
        self.PI = I - PI
        
    
    def setup(self, Q : np.array, R : np.array, lam_g1=None, lam_g2=None, lam_y=None) -> None:
       
        """
        Set up controller constraints and cost function.
        Also used online during sim to update u_ini, y_ini, reference and regularizers
        args:
            ref = reference signal
            u_ini = initial input trajectory
            y_ini = initial output trajectory
            lam_g1, lam_g2 = regularization params for nonlinear systems
            lam_y = regularization params for stochastic systems
        """

        self.lam_y = lam_y
        self.lam_g1 = lam_g1
        self.lam_g2 = lam_g2
        self.Q = np.kron(np.eye(self.N), Q)
        self.R = np.kron(np.eye(self.N), R)
        
        self.cost = cp.quad_form(self.y-self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u-self.u_ref, cp.psd_wrap(self.R))


        if self.lam_y != None:
            self.cost += cp.norm(self.sig_y, 1)*self.lam_y
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini + self.sig_y,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower
            ]
        else:
            self.constraints = [
                self.Up@self.g == self.u_ini,
                self.Yp@self.g == self.y_ini,
                self.Uf@self.g == self.u,
                self.Yf@self.g == self.y,
                self.u <= self.u_upper, self.u >= self.u_lower,
                self.y <= self.y_upper, self.y >= self.y_lower
            ]

        if self.lam_g1 != None:
            self.cost += cp.sum_squares(self.PI@self.g)*lam_g1 
        if self.lam_g2 != None:
            self.cost += cp.norm(self.g, 1)*lam_g2
        assert self.cost.is_dpp

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self

    def solve(self, y_ref, u_ref, u_ini, y_ini, verbose=False, solver=cp.MOSEK) -> np.ndarray:
        
        """
        Call once the controller is set up with relevenat parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, usually use MOSEK
            verbose = bool for printing status of solver
        """

        # prob = cp.Problem(cp.Minimize(self.cost), self.constraints)
        # assert prob.is_dpp()
        # assert prob.is_dcp()
        self.y_ref.value = y_ref
        self.u_ref.value = u_ref
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)
        action = self.problem.variables()[1].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs
    
class npMPC:

    def __init__(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray, 
                 N: int, u_constraints: np.ndarray, y_constraints: np.ndarray) -> None:
        
        self.N = N
        self.p = B.shape[0] 
        self.N = N
        self.m = B.shape[1]
        self.y_lower = y_constraints[0]
        self.y_upper = y_constraints[1]
        self.u_lower= u_constraints[0]
        self.u_upper = u_constraints[1]
        self.A = cp.Parameter(A.shape)
        self.B = cp.Parameter(B.shape)
        self.Q = Q
        self.R = R
        self.A.value = A
        self.B.value = B
        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N*self.m)
        self.y = cp.Variable(self.N*self.p)
        self.y_ref = cp.Parameter((self.N*self.p,))
        self.u_ref = cp.Parameter((self.N*self.m,))
        self.y_ini = cp.Parameter(self.p)

    def setup(self):

        self.Q = np.kron(np.eye(self.N), self.Q)
        self.R = np.kron(np.eye(self.N), self.R)
        self.cost = cp.quad_form(self.y-self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u-self.u_ref, cp.psd_wrap(self.R))
        
        self.constraints = [
            self.y[:self.p] == self.y_ini,
            self.u <= self.u_upper, self.u >= self.u_lower,
            self.y <= self.y_upper, self.y >= self.y_lower
        ]

        for i in range(1,self.N):
            self.constraints.append(
                self.y[self.p*i:self.p*(i+1)] == self.A@self.y[self.p*(i-1):self.p*i] + self.B@self.u[self.m*(i-1):self.m*i]
            )

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self
    
    def solve(self, y_ref, u_ref, y_ini, u_ini=None, verbose=False, solver=cp.OSQP) -> np.ndarray:
        self.y_ref.value = y_ref
        self.u_ref.value = u_ref
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)
        action = self.problem.variables()[1].value[:self.m]
        obs = self.problem.variables()[0].value # For imitation loss
        return action, obs






class npDeePC:
    """
    Vanilla regularized DeePC module
    """

    def __init__(self, ud: np.ndarray, yd: np.ndarray, y_constraints: Tuple[np.ndarray],
                 u_constraints: Tuple[np.ndarray],
                 N: int, Tini: int) -> None:
        """
        Initialise variables
        args:
            ud = Input signal data
            yd = output signal data
            N = prediction horizon
            n = dimension of system
            p = output signal dimension
            m = input signal dimension
        """

        self.T = ud.shape[0]  # Length of data
        self.Tini = Tini  # Length of initial trajectory

        self.N = N  # Prediction horizon
        self.p = yd.shape[1]  # Output signal dimension
        self.m = ud.shape[1]  # Input signal dimension
        self.y_lower = y_constraints[0]  # Output constraints
        self.y_upper = y_constraints[1]  # Output constraints
        self.u_lower = u_constraints[0]  # Input constraints
        self.u_upper = u_constraints[1]  # Input constraints

        # Check for full row rank
        H = block_hankel(w=ud.reshape((self.m * self.T,)), L=Tini + N, d=self.m)
        rank = np.linalg.matrix_rank(H)
        if rank != H.shape[0]:
            raise ValueError('Data is not persistently exciting')

        # Construct data matrices
        U = block_hankel(w=ud.reshape((self.m * self.T,)), L=Tini + N, d=self.m)  # Input hankel matrix w input as column vector
        Y = block_hankel(w=yd.reshape((self.p * self.T,)), L=Tini + N, d=self.p)  # Output hankel matrix w input as column vector

        self.Up = U[0:self.m * Tini, :]  # Past input trajectory
        self.Yp = Y[0:self.p * Tini, :]  # Past output trajectory
        self.Uf = U[Tini * self.m:, :]  # Future input trajectory
        self.Yf = Y[Tini * self.p:, :]  # Future output trajectory

        # Initialise Optimisation variables and parameters
        self.u = cp.Variable(self.N * self.m)  # Input trajectory
        self.g = cp.Variable(self.T - self.Tini - self.N + 1)  # g vector
        self.y = cp.Variable(self.N * self.p)  # Output trajectory

        self.y_ref = cp.Parameter((self.N * self.p,))  # Reference output trajectory
        self.u_ini = cp.Parameter(self.Tini * self.m)  # Initial input trajectory to be fed in later
        self.y_ini = cp.Parameter(self.Tini * self.p)  # Initial output trajectory to be fed in later

    def setup(self, Q: np.array, R: np.array) -> None:
        """
        Set up controller constraints and cost function.
        Also used online during sim to update u_ini, y_ini, reference and regularizers
        args:
            ref = reference signal
            u_ini = initial input trajectory
            y_ini = initial output trajectory
            lam_g1, lam_g2 = regularization params for nonlinear systems
            lam_y = regularization params for stochastic systems
        """

        self.Q = np.kron(np.eye(self.N), Q)
        self.R = np.kron(np.eye(self.N), R)

        self.cost = cp.quad_form(self.y - self.y_ref, cp.psd_wrap(self.Q)) + cp.quad_form(self.u, cp.psd_wrap(self.R))

        self.constraints = [
            self.Up @ self.g == self.u_ini,
            self.Yp @ self.g == self.y_ini,
            self.Uf @ self.g == self.u,
            self.Yf @ self.g == self.y,
            self.u <= self.u_upper, self.u >= self.u_lower,
            self.y <= self.y_upper, self.y >= self.y_lower
        ]

        assert self.cost.is_dpp

        self.problem = cp.Problem(cp.Minimize(self.cost), self.constraints)
        return self

    def solve(self, y_ref, u_ini, y_ini, verbose=False, solver=cp.MOSEK) -> np.ndarray:
        """
        Call once the controller is set up with relevant parameters.
        Returns the first action of input sequence.
        args:
            solver = cvxpy solver, usually use MOSEK
            verbose = bool for printing status of solver
        """

        self.y_ref.value = y_ref
        self.u_ini.value = u_ini
        self.y_ini.value = y_ini
        self.problem.solve(solver=solver, verbose=verbose)
        action = self.problem.variables()[1].value[:self.m]
        return action


# Below will be running this problem for the linear double integrator system

# double_integrator = DoubleIntegrator()

# # first get the trajectories. 
# T = 70  # length of trajectory
# u_min = -3
# u_max = 3
# dt = 0.02
# T_ini = 4
# N = 20
# p = 2
# m = 1
# total_simulation_time = 10
# n_steps = int(total_simulation_time / dt)

# u_traj = np.random.uniform(u_min, u_max, T) # Generate random control inputs
# initial_state = np.array([0, 0]) # Initial state of the system
# y_traj = double_integrator.generate_trajectory(initial_state, u_traj)

# y_ref = double_integrator.generate_reference_trajectory(0, 1, N)
# # print(y_ref)

# # print(u_traj, y_traj)


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

# deepc = npDeePC(u_traj, y_traj, y_constraints=y_constraints, u_constraints=u_constraints, N=N, Tini=T_ini, n=2, p=p,
#                 m=m)

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

# states = [y_ini[-p:]]
# controls = []

# # print(u_ini, y_ini)

# for t in range(n_steps):
#     new_u = deepc.solve(y_ref, u_ini, y_ini)

#     u_ini = np.concatenate((u_ini[1:], new_u))
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
#              with_video=False, figure_dir="figures", figure_name='Linear DeePC', linear=True)
