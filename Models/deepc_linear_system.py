import numpy as np
from scipy import signal as scipysig
from Models.deepc_system import Data, SystemBase

class LinearSystem(SystemBase):
    """
    Represents a dynamical system that can be simulated
    """
    def __init__(self, sys: scipysig.StateSpace, x0: np.array, m: int, measurement_noise_std: float = 0):
        """
        :param sys: a linear system
        :param x0: initial state
        """
        super().__init__(x0, m=m, measurement_noise_std=measurement_noise_std)
        self.sys = sys

    def step(self, u: np.ndarray) -> Data:
        T = len(u)
        if T > 1:
            # If u is a signal of length > 1 use dlsim for quicker computation
            t, y, x0 = scipysig.dlsim(self.sys, u, t = np.arange(T) * self.sys.dt, x0 = self.x0)
            self.x0 = x0[-1]
        else:
            y = self.sys.C @ self.x0
            x = self.x0
            self.x0 = self.sys.A @ self.x0.flatten() + self.sys.B @ u.flatten()
            
        return y, x
        