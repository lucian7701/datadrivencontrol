import numpy as np

class DoubleIntegrator:
    def __init__(self, dt=0.02):
        # System parameters
        self.dt = dt

        # Continuous-time system matrices
        self.A = np.array([
            [0, 1],
            [0, 0]
        ])

        self.B = np.array([
            [0],
            [1]
        ])

        # Discretized system matrices using series expansion
        I = np.eye(self.A.shape[0])
        self.Ad = I + self.A*dt + (self.A @ self.A)*(dt**2)/2
        self.Bd = (I*dt + (self.A*dt**2)/2) @ self.B

        # print("A_d:")
        # print(self.Ad)
        # print("B_d:")
        # print(self.Bd)


    def step(self, state, control_input):
        """
        Update the state of the system given a control input.
        
        :param state: The current state of the system (2-dimensional vector)
        :param control_input: The control input to the system (scalar)
        :return: The next state of the system (2-dimensional vector)
        """
        next_state = self.Ad @ state + self.Bd.flatten() * control_input
        return next_state

    def generate_trajectory(self, initial_state, control_inputs):
        """
        Generate a trajectory given an initial state and a sequence of control inputs.
        
        :param initial_state: The initial state of the system (2-dimensional vector, np.array)
        :param control_inputs: A sequence of control inputs (list or array of scalars)
        :return: A trajectory of states (array of 2-dimensional vectors)
        """
        num_steps = control_inputs.shape[0]
        trajectory = np.zeros((num_steps, initial_state.shape[0]))
        state = initial_state
        for i, u in enumerate(control_inputs):
            state = self.step(state, u)
            trajectory[i] = state
        return trajectory
    

    # Generate reference trajectory
    def generate_reference_trajectory(self, start, end, steps):
        positions = np.linspace(start, end, steps)
        velocities = np.zeros(steps)
        reference_trajectory = np.vstack((positions, velocities)).T
        return reference_trajectory

# # Example usage
# double_integrator = DoubleIntegrator()
# initial_state = np.array([0, 0])
# control_inputs = np.random.uniform(-1, 1, 100)  # Generate random control inputs
# trajectory = double_integrator.generate_trajectory(initial_state, control_inputs)

# print("Generated trajectory shape:", trajectory.shape)
# print("First few states in the trajectory:\n", trajectory[:5])
