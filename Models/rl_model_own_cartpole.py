import numpy as np
from scipy.integrate import solve_ivp
import math

class CustomContinuousCartPoleEnv:
    def __init__(self, length=0.5, masspole=0.1, gravity=9.8, tau=0.02):
        self.length = length
        self.masspole = masspole
        self.gravity = gravity
        self.tau = tau  # Time step for simulation
        self.total_mass = self.masspole + 1.0  # Assume cart mass = 1.0
        self.polemass_length = self.masspole * self.length
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.state = None
        self.steps_beyond_done = None
        self.reward_range = (0, 1)
        self.target_positions = np.array([0, 0, 0, 0])

        self.reset()
    
    def dynamics(self, t, y, force):
        x, x_dot, theta, theta_dot = y
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        return [x_dot, xacc, theta_dot, thetaacc]
    
    def step(self, force, eval=False):
        assert isinstance(force, float), "Force must be a float."
        
        # Integrate the equations of motion
        sol = solve_ivp(fun=self.dynamics, t_span=[0, self.tau], y0=self.state, args=(force,))
        self.state = sol.y[:, -1]  # Get the final state after integration
        
        x, x_dot, theta, theta_dot = self.state

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        if eval:
            done = False
            
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}, {}

    def reset(self):
        self.state = np.array([
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05),
            np.random.uniform(low=-0.05, high=0.05)
        ])
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32), {}


