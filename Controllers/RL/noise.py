"""
For reference this code is based on the following repository:
https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code/tree/master/DDPG
"""
import numpy as np

class OUActionNoise():
    def __init__(self, mu, sigma=0.5, theta=0.15, dt=0.1, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0 
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x

        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

