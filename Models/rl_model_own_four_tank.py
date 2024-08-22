import numpy as np
from scipy.integrate import solve_ivp

class CustomFourTankEnv:
    def __init__(self, target_positions=np.array([14.0, 14.0, 14.2, 21.3]), dt=3, control_penalty=0.001, position_penalty=1.0, k=1):
        
        # Environment parameters
        self.target_positions = target_positions  # Desired position (reference)
        self.dt = dt  # Time step for simulation
        self.control_penalty = control_penalty  # Penalty factor for control effort
        self.position_penalty = position_penalty  # Penalty factor for position error
        # State: [position, velocity]
        self.state = None
        self.k = k
        # Reward range (can be customized based on your design)
        self.reward_range = (-float('inf'), float('inf'))


    def dynamics(self, t, y, u):

        # Constants
        g = 981
        a1 = 0.233
        a2 = 0.242
        a3 = 0.127
        a4 = 0.127
        A1 = 50.27
        A2 = 50.27
        A3 = 28.27
        A4 = 28.27
        gamma1 = 0.4
        gamma2 = 0.4

        # State derivatives
        dxdt = [
                (-a1 / A1) * np.sqrt(2 * g * y[0] + 1e-3) + (a3 / A1)
                    * np.sqrt(2 * g * y[2] + 1e-3) + (gamma1 / A1) * u[0],
                (-a2 / A2) * np.sqrt(2 * g * y[1] + 1e-3) + a4 / A2
                    * np.sqrt(2 * g * y[3]+ 1e-3) + (gamma2 / A2) * u[1],
                (-a3 / A3) * np.sqrt(2 * g * y[2] + 1e-3) + (1 - gamma2) / A3 * u[1],
                    (-a4 / A4) * np.sqrt(2 * g * y[3] + 1e-3) + (1 - gamma1) / A4 * u[0]
        ]

        return dxdt


    def step(self, action):
        assert isinstance(action, np.ndarray) and action.shape == (2,), "Action must be a 2-element array."

        # Integrate the dynamics over the time step
        sol = solve_ivp(fun=self.dynamics, t_span=[0, self.dt], y0=self.state, args=(action,))
        self.state = sol.y[:, -1]  # Get the final state after integration

        # ADD NOISE HERE PERHAPS? this would be for when testing the algorithm to noise

        # Calculate the error (difference between target position and current position)
        position_error = abs(self.target_positions - self.state)

        # Reward function: Positive reward based on proximity to target
        # The closer the position is to the target, the higher the reward
        reward = np.sum(np.exp(-self.k*position_error))


        # Small penalty for control effort to encourage efficiency
        control_penalty = np.sum(self.control_penalty * action)

        # The final reward combines the proximity reward with a small penalty for control effort
        reward = reward - control_penalty

        # Ensure the reward is non-negative
        reward = max(reward, 0)

        # Check if the episode is done (position error is too large)
        done = np.any(abs(position_error) > 7.0)
        
        return np.array(self.state, dtype=np.float32), reward, done, {}, {}

    def reset(self):
        # Reset the state to initial conditions (position and velocity close to zero)
        self.state = np.array([
            np.random.uniform(low=7.95, high=8.05),  # Initial position
            np.random.uniform(low=9.95, high=10.05),   # Initial velocity
            np.random.uniform(low=7.95, high=8.05),   # Initial velocity
            np.random.uniform(low=18.95, high=19.05)   # Initial velocity
        ])

        return np.array(self.state, dtype=np.float32), {}
        
        # x0 = np.array([8, 10, 8, 19]) 