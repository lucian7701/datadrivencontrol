from Analysis.state_control_plot import plot_state_control_util
import numpy as np

class Analyser: 

    def __init__(self, states, controls, reference_trajectory, T_ini=0):
        self.states = states
        self.controls = controls
        self.reference_trajectory = reference_trajectory
        self.T_ini = T_ini

    
    def plot_state_control(self, dt, state_labels, control_labels, ref_labels):
        plot_state_control_util(self.states, self.controls, dt, state_labels, control_labels, reference_trajectory=self.reference_trajectory, T_ini=self.T_ini, ref_labels=ref_labels)


    def absolute_error(self) -> np.array: 
        return np.abs(self.states[self.T_ini:] - self.reference_trajectory)
    
    def total_absolute_error_by_state(self) -> np.array: 
        return np.sum(self.absolute_error(), axis=0)
    
    def total_absolute_error(self) -> float: 
        return np.sum(self.absolute_error())

    def absolute_control(self) -> np.array: 
        return np.abs(self.controls[self.T_ini:])
    
    def total_absolute_control_by_input(self) -> np.array:
        return np.sum(self.absolute_control(), axis=0)
    
    def total_absolute_control(self) -> float:
        return np.sum(self.absolute_control())
