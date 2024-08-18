import numpy as np
from scipy.integrate import solve_ivp
from Models.deepc_model_l import System

class CustomDoubleIntegratorEnv:
    def __init__(self, target_position=7.0, dt=0.02, control_penalty=0.01, position_penalty=1.0):
        # Environment parameters
        self.target_position = target_position  # Desired position (reference)
        self.dt = dt  # Time step for simulation
        self.control_penalty = control_penalty  # Penalty factor for control effort
        self.position_penalty = position_penalty  # Penalty factor for position error

        # State: [position, velocity]
        self.state = None
        
        # Reward range (can be customized based on your design)
        self.reward_range = (-float('inf'), float('inf'))

    def dynamics(self, t, y, u):
        # Double integrator dynamics: y'' = u
        position, velocity = y
        acceleration = u
        return [velocity, acceleration]
    
    def step(self, action):
        assert isinstance(action, float), "Action must be a float."

        # Integrate the dynamics over the time step
        sol = solve_ivp(fun=self.dynamics, t_span=[0, self.dt], y0=self.state, args=(action,))
        self.state = sol.y[:, -1]  # Get the final state after integration

        # ADD NOISE HERE PERHAPS?

        # Calculate the error (difference between target position and current position)
        position_error = self.target_position - self.state[0]

        # Reward function: Negative of squared error and control effort
        reward = -self.position_penalty*position_error**2 - self.control_penalty * action**2

        # Check if the episode is done (position is close to target)
        done = bool(abs(position_error) < 0.1 or
                     abs(position_error) > 20.0)
        
        return np.array(self.state[0], dtype=np.float32), reward, done, {}, {}

    def reset(self):
        # Reset the state to initial conditions (position and velocity close to zero)
        self.state = np.array([
            np.random.uniform(low=-0.05, high=0.05),  # Initial position
            np.random.uniform(low=-0.05, high=0.05)   # Initial velocity
        ])

        return np.array(self.state[0], dtype=np.float32), {}


