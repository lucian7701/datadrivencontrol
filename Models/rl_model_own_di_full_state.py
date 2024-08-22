import numpy as np
from scipy.integrate import solve_ivp

class CustomDoubleIntegratorFullEnv:
    def __init__(self, target_position=7.0, dt=0.02, control_penalty=0.001, position_penalty=1.0, k=1):
        
        # Environment parameters
        self.target_position = target_position  # Desired position (reference)
        self.dt = dt  # Time step for simulation
        self.control_penalty = control_penalty  # Penalty factor for control effort
        self.position_penalty = position_penalty  # Penalty factor for position error
        # State: [position, velocity]
        self.state = None
        
        # Reward range (can be customized based on your design)
        self.reward_range = (-float('inf'), float('inf'))

        self.k = k


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

        # ADD NOISE HERE PERHAPS? this would be for when testing the algorithm to noise

        # Calculate the error (difference between target position and current position)
        
        position_error = self.target_position - self.state[0]

        # Reward function: Positive reward based on proximity to target
        # The closer the position is to the target, the higher the reward
        # proximity_reward = self.position_penalty / (1 + abs(position_error))
        
        # TODO change this back, if you do not want exponential reward
        proximity_reward = np.exp(-self.k * abs(position_error))

        # Small penalty for control effort to encourage efficiency
        control_penalty = self.control_penalty * action

        # The final reward combines the proximity reward with a small penalty for control effort
        reward = proximity_reward - control_penalty

        # Ensure the reward is non-negative
        reward = max(reward, 0)

        # Check if the episode is done (position error is too large)
        done = bool(abs(position_error) > 7.0)
        
        return np.array(self.state, dtype=np.float32), reward, done, {}, {}

    def reset(self):
        # Reset the state to initial conditions (position and velocity close to zero)
        self.state = np.array([
            np.random.uniform(low=-0.05, high=0.05),  # Initial position
            np.random.uniform(low=-0.05, high=0.05)   # Initial velocity
        ])

        return np.array(self.state, dtype=np.float32), {}
