from Models.deepc_system import Data

from Controllers.DeePC.DeePC import DeePC
import numpy as np
from Analysis.state_control_reference import plot_results
from Analysis.Analyser import Analyser
import os


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
        self.y_ref_original = y_ref if y_ref is not None else np.ones((self.N, self.p))
        self.u_ref = u_ref if u_ref is not None else np.zeros((self.N, self.m))
        self.data_ini = data_ini if data_ini is not None else Data(u=np.zeros((int(T_ini), int(m))), y=np.ones((int(T_ini), int(p)))*0)

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
        plot_results(states, controls, self.dt, reference_trajectory=y_ref, T_ini=self.T_ini, state_labels=None, control_labels=None)

    def run_eval(self, state_labels=None, control_labels=None, ref_labels=None, plot: bool = True, filename = None):

        data = self.sys.get_all_samples()
        states, controls = data.y, data.u

        analyser = Analyser(states=states, controls=controls, reference_trajectory=np.array([self.y_ref_original[0]]).reshape(1,-1))
        if plot:
            analyser.plot_state_control(self.dt, state_labels, control_labels, ref_labels)


        total_absolute_error = analyser.total_absolute_error()
        total_absolute_control = analyser.total_absolute_control()
        total_absolute_error_by_state = analyser.total_absolute_error_by_state()
        total_absolute_control_by_input = analyser.total_absolute_control_by_input()
    

        if filename is not None:
            file_path = os.path.join(os.getcwd(), f'performance/{filename}.npz')
            try:
                previous_data = np.load(file_path)
                errors = list(previous_data['errors'])
                controls = list(previous_data['controls'])
                errors_by_state = list(previous_data['errors_by_state'])
                controls_by_input = list(previous_data['controls_by_input'])
            except FileNotFoundError:
                # Initialize lists if file doesn't exist
                errors = []
                controls = []
                errors_by_state = []
                controls_by_input = []

            errors.append(total_absolute_error)
            controls.append(total_absolute_control)
            errors_by_state.append(total_absolute_error_by_state)
            controls_by_input.append(total_absolute_control_by_input)

            np.savez(file_path, errors=errors, controls=controls, errors_by_state=errors_by_state, controls_by_input=controls_by_input)


        print("Total absolute error:", total_absolute_error)
        print("Total absolute control:", total_absolute_control)

        print("Total absolute error by state:", total_absolute_error_by_state)
        print("Total absolute control by input:", total_absolute_control_by_input)
