"""https://github.com/rssalessio/PyDeePC/blob/main/README.md"""

# Model nonlinear for DeePC

import numpy as np
from typing import Optional
from Models.deepc_model_l import Data
from Models.gpmpc_model import Model
from scipy.integrate import solve_ivp


class System(object):
    """
    Represents a dynamical system that can be simulated
    """
    def __init__(self, ode, m, Nx, dt, x0: Optional[np.ndarray] = None, Ny = None):
        """
        :param x0: initial state
        """
        self.Nx = Nx
        self.m = m
        self.dt = dt
        self.ode = ode
        self.Ny = Ny if Ny is not None else Nx
        
        self.x0 = x0 if x0 is not None else np.zeros(self.Nx)
        self.u = None
        self.y = None
        self.x = x0 if x0 is not None else np.zeros(self.Nx)


    def apply_input(self, u: np.ndarray, noise_std: float = 0) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        T = len(u)
        
        
        t_span = (0, self.dt)

        for i in range(T):

            # Calculate output here if needed
            y = self.x0 
            
            solution = solve_ivp(self.ode, t_span, self.x0, args=(u[i],), method='RK45', t_eval=[self.dt])
            
            # Extract the state at the end of the time step
            self.x0 = solution.y[:, -1]
            self.x = np.vstack([self.x, self.x0]) if self.x is not None else self.x0
            self.y = np.vstack([self.y, y]) if self.y is not None else y
            self.u = np.vstack([self.u, u[i,]]) if self.u is not None else u[i,]

        # This is adding measurement noise
        self.y = self.y + noise_std * np.random.normal(size = (T, self.Ny))


        return Data(self.u, self.y)
    
    def generate_bounded_data(self, T: int, u_min: np.ndarray, u_max: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> Data:
        

        # Initial state
        x_current = self.x0
        u_sequence = None
        x_sequence = None

        for t in range(T):

            x_next = None
            
            while x_next is None or np.any(x_next < x_min) or np.any(x_next > x_max):

                # Generate random input within the bounds
                u_random = np.random.uniform(low=u_min, high=u_max)

                # Simulate system for one step with current input
                t_span = (0, self.dt)
                solution = solve_ivp(self.ode, t_span, x_current, args=(u_random,), method='RK45', t_eval=[self.dt])
                x_next = solution.y[:, -1]
                
            
            u_sequence = np.vstack([u_sequence, u_random]) if u_sequence is not None else u_random
            x_sequence = np.vstack([x_sequence, x_next]) if x_sequence is not None else x_next
            x_current = x_next

        return Data(u_sequence, x_sequence)


    def get_last_n_samples(self, n: int) -> Data:
        """
        Returns the last n samples
        :param n: integer value
        """
        assert self.u.shape[0] >= n, 'Not enough samples are available'
        return Data(self.u[-n:], self.y[-n:])

    def get_all_samples(self) -> Data:
        """
        Returns all samples
        """
        return Data(self.u, self.y)
    
    def reset(self, data_ini: Optional[Data] = None, x0: Optional[np.ndarray] = None):
        """
        Reset initial state and collected data
        """
        self.u = None if data_ini is None else data_ini.u
        self.y = None if data_ini is None else data_ini.y
        self.x0 = x0 if x0 is not None else self.x0
