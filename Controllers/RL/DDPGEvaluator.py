import numpy as np
import matplotlib.pyplot as plt
from Controllers.RL.ddpg_torch import Agent

class DDPGEvaluator:

    def __init__(self, env, input_dims=[4], n_actions=1):
        self.env = env
        self.agent = Agent(alpha=0.0001, beta=0.001, 
                           input_dims=input_dims, tau=0.005,
                           batch_size=64, fc1_dims=400, fc2_dims=300, 
                           n_actions=n_actions)
        self.agent.load_models()  # Load trained models

    def run_eval(self):
        observation, _ = self.env.reset()
        done = False
        visited_states = []
        control_inputs = []
        rewards = []

        while not done:
            action = self.agent.choose_action_eval(observation)
            control_inputs.append(action)
            visited_states.append(observation)

            observation_, reward, done, info, _ = self.env.step(action.item())
            rewards.append(reward)
            observation = observation_

        visited_states = np.array(visited_states)
        control_inputs = np.array(control_inputs)
        rewards = np.array(rewards)

        # Ensure visited_states is 2-dimensional before plotting
        if visited_states.ndim == 1:
            visited_states = np.expand_dims(visited_states, axis=1)

        self.plot_results(visited_states, control_inputs, rewards)

    def plot_results(self, states, actions, rewards):
        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        if states.shape[1] > 1:  # Check if states has more than one dimension
            plt.plot(states[:, 0], label='Position')
            plt.plot(states[:, 1], label='Velocity')
        else:
            plt.plot(states, label='State')

        plt.title('State Variables Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('State Values')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(actions, label='Control Input (Action)', color='g')
        plt.title('Control Inputs Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Control Input')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(rewards, label='Reward', color='r')
        plt.title('Rewards Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Reward')
        plt.legend()

        plt.tight_layout()
        plt.show()