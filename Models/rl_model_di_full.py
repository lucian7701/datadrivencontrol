import numpy as np
from scipy.integrate import solve_ivp

class CustomDoubleIntegratorFullEnv:
    def __init__(self, target_positions=np.array([7, 0]), dt=0.1, Q=np.array([[10, 0], [0, 1]]), R=0.1, k=0.01, reward_type='exponential'):
        
        # Environment parameters
        self.target_positions = target_positions  # Target position for the system 
        self.dt = dt  # Time step for simulation
        self.state = None
        
        # Reward range (can be customized based on your design)
        self.reward_range = (-float('inf'), float('inf'))
        self.k = k

        self.Q = Q
        self.R = R

        self.reward_type = reward_type


    def dynamics(self, t, y, u):
        # Double integrator dynamics: y'' = u
        position, velocity = y
        acceleration = u

        return [velocity, acceleration]
    

    def step(self, action, eval=False):
        assert isinstance(action, float), "Action must be a float."

        # Integrate the dynamics over the time step
        sol = solve_ivp(fun=self.dynamics, t_span=[0, self.dt], y0=self.state, args=(action,))
        self.state = sol.y[:, -1]  # Get the final state after integration

        # ADD NOISE HERE PERHAPS? this would be for when testing the algorithm to noise

        position_error = self.target_positions - self.state
        proximity_error = np.dot(np.dot(position_error.T, self.Q), position_error)
        # print(proximity_error)
        control_effort = self.R * action**2
        
        if self.reward_type == 'exponential':
            reward = self.exponential_reward(proximity_error, control_effort)
            if eval:
                failure_condition = False
            else:
                failure_condition = abs(position_error[0]) > self.target_positions[0] + 3
            success_condition = False
        
        elif self.reward_type == 'negative_quadratic':
            reward = self.negative_quadratic_reward(proximity_error, control_effort)
            failure_condition = abs(self.state[0]) > self.target_positions[0] + 20
            success_condition = np.linalg.norm(position_error) < 0.01

        done = failure_condition or success_condition
        
        return np.array(self.state, dtype=np.float32), reward, done, {}, {}


    def reset(self):
        # Reset the state to initial conditions (position and velocity close to zero)
        self.state = np.array([
            np.random.uniform(low=-0.05, high=0.05),  # Initial position
            np.random.uniform(low=-0.05, high=0.05)   # Initial velocity
        ])

        return np.array(self.state, dtype=np.float32), {}


    def exponential_reward(self, proximity_error, control_effort):
        reward = np.exp(-self.k * (proximity_error))
        return reward * 100


    def negative_quadratic_reward(self, proximity_error, control_effort):
        reward = -proximity_error - control_effort
       
        return reward
    