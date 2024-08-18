import numpy as np
from typing import Optional, NamedTuple, Union
from abc import ABC, abstractmethod


class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param y: output data
    """
    u: np.ndarray
    y: np.ndarray


class SystemBase(ABC):

    """
    Represents a dynamical system that can be simulated
    """
    
    def __init__(self, x0: np.array, m, measurement_noise_std: float):
        """
        :param x0: initial state
        """
        self.m = m
        self.original_x0 = x0
        self.x0 = x0 
        self.u = None
        self.y = None
        self.x = None
        self.measurement_noise_std = measurement_noise_std

    @abstractmethod
    def step(self, u: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        """
        Applies the input to the system. 
        Returns the output state and the state of the system.
        """
        pass


    def apply_input(self, u: np.array) -> Data:
        """
        Applies an input signal to the system.
        :param u: input signal. Needs to be of shape T x M, where T is the batch size and
                  M is the number of features
        :param noise_std: standard deviation of the measurement noise
        :return: tuple that contains the (input,output) of the system
        """
        
        y, x = self.step(u)

        # This is adding measurement noise
        if y.ndim == 1:
        # 1D array
            noise = np.random.normal(scale=self.measurement_noise_std, size=y.shape[0])
        elif y.ndim == 2:
            # 2D array
            noise = np.random.normal(scale=self.measurement_noise_std, size=(y.shape[0], y.shape[1]))

        y = y + noise

        self.u = np.vstack([self.u, u]) if self.u is not None else u
        self.y = np.vstack([self.y, y]) if self.y is not None else y
        self.x = np.vstack([self.x, x]) if self.x is not None else x

        return Data(self.u, self.y)
        

    def generate_training_data(self, T: int, u_min: np.ndarray, u_max: np.ndarray, x_min: np.ndarray=None, x_max: np.ndarray=None) -> Data:
        
        u_sequence = None
        y_sequence = None

        for t in range(T):
            while True:
                # Generate random input within the bounds
                u_random = np.random.uniform(low=u_min, high=u_max, size=(1, self.m))
                # Simulate system for one step with current input
                y, x = self.step(u_random)

                if (x_min is None or np.all(self.x0 >= x_min)) and (x_max is None or np.all(self.x0 <= x_max)):
                    break

            u_sequence = np.vstack([u_sequence, u_random]) if u_sequence is not None else u_random
            y_sequence = np.vstack([y_sequence, y]) if y_sequence is not None else y

        return Data(u_sequence, y_sequence)


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
        self.x0 = self.original_x0 if x0 is None else x0
