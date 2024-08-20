from Models.deepc_model_l import Data

from Controllers.DeePC.DeePC import DeePC
import numpy as np
from Analysis.state_control_reference import plot_results


class DeePCExecutor:

    def __init__(self, T: int, N: int, m: int, p: int, T_ini: int, total_simulation_time: int, 
                 dt: float, sys, Q: np.array, R: np.array, training_data: Data, lam_g1: float = None, lam_g2: float = None, lam_y: float = None,
                 u_min: np.array = None, u_max: np.array = None,
                 y_min: np.array = None, y_max: np.array = None, 
                 y_ref: np.array=None, u_ref: np.array=None, data_ini: Data=None):
        """
        Initialize the DeePCExecutor class.

        Parameters:
        - T: int - Prediction horizon length.
        - N: int - Number of control intervals.
        - m: int - Number of inputs.
        - p: int - Number of outputs.
        - T_ini: int - Length of initial data used for system identification.
        - total_simulation_time: int - Total simulation time.
        - dt: float - Time step.
        - sys: System - System model.
        - Q: np.array - State cost matrix.
        - R: np.array - Control cost matrix.
        - u_min: np.array - Minimum input values.
        - u_max: np.array - Maximum input values.
        - y_min: np.array - Minimum output values.
        - y_max: np.array - Maximum output values.
        - y_ref: np.array - Reference output values. Default is None.
        - u_ref: np.array - Reference input values. Default is None.
        - data_ini: Data - Initial data for system identification. Default is None.
        """

        self.T = T
        self.N = N
        self.m = m
        self.p = p
        self.u_min = u_min
        self.u_max = u_max 
        self.y_min = y_min 
        self.y_max = y_max 

        self.T_ini = T_ini
        self.total_simulation_time = total_simulation_time
        self.dt = dt
        self.sys = sys
        self.Q = Q
        self.R = R
        self.lam_g1 = lam_g1
        self.lam_g2 = lam_g2
        self.lam_y = lam_y
        self.y_ref = y_ref if y_ref is not None else np.ones((self.N, self.p))
        self.u_ref = u_ref if u_ref is not None else np.zeros((self.N, self.m))
        self.data_ini = data_ini if data_ini is not None else Data(u=np.zeros((int(T_ini), int(m))), y=np.zeros((int(T_ini), int(p))))

        self.n_steps = int(total_simulation_time // dt)

        self.training_data = training_data


        # Extend the constraints over the prediction horizon using np.kron
        self.y_upper = np.kron(np.ones(self.N), self.y_max)
        self.y_lower = np.kron(np.ones(self.N), self.y_min)
        self.u_upper = np.kron(np.ones(self.N), self.u_max)
        self.u_lower = np.kron(np.ones(self.N), self.u_min)

        # Combine into tuples for constraints
        self.y_constraints = (self.y_lower, self.y_upper)
        self.u_constraints = (self.u_lower, self.u_upper)

        # Initialize DeePC controller
        self.deepc = DeePC(self.training_data.u, 
                           self.training_data.y, 
                           y_constraints=self.y_constraints, 
                           u_constraints=self.u_constraints, 
                           N=N, Tini=T_ini, p=p, m=m)
        
        self.deepc.setup(self.Q, self.R, lam_g1=self.lam_g1, lam_g2=self.lam_g2, lam_y=self.lam_y)

        # Reshape reference values
        self.y_ref = self.y_ref.reshape(N * p, )
        self.u_ref = self.u_ref.reshape(N * m, )

        # Reset system with initial data
        self.sys.reset(data_ini = self.data_ini)


    def run(self):

        # Note that adjustments would need to be made if y_ref was not just constant. 
        for t in range(self.n_steps):

            u_ini = self.sys.get_last_n_samples(self.T_ini).u.reshape(self.T_ini*self.m, )
            y_ini = self.sys.get_last_n_samples(self.T_ini).y.reshape(self.T_ini*self.p, )
            
            new_u, _ = self.deepc.solve(self.y_ref, self.u_ref, u_ini, y_ini)
            new_u = new_u.reshape(-1, self.m)

            self.sys.apply_input(new_u)
            self.sys.store_ref(self.y_ref[:self.p].T)


    def plot(self):

        data = self.sys.get_all_samples()
        states, controls = data.y, data.u
        y_ref = self.sys.get_ref()
        plot_results(states, controls, self.total_simulation_time+self.T_ini, title='DeePC Controller', reference_trajectory=y_ref, T_ini=self.T_ini)
