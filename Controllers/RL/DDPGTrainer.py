import numpy as np
from Controllers.RL.ddpg_torch import Agent
import matplotlib.pyplot as plt


class DDPGTrainer:

    def __init__(self, env, input_dims=[4], n_actions=1, load_checkpoint=False):
        """
        env: gym environment
        """
        self.env = env
        self.agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=input_dims, tau=0.005,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=n_actions)
        

        self.n_games = 2500
        self.max_steps_per_episode = 1000
        self.best_score = self.env.reward_range[0]
        self.score_history = []

        if load_checkpoint:
            self.agent.load_models()

    
    def run(self):

        for i in range(self.n_games):
            observation, _ = self.env.reset()
            done = False
            score = 0
            self.agent.noise.reset()
            step_count = 0
            while not done and step_count < self.max_steps_per_episode:
                action = self.agent.choose_action(observation)
                observation_, reward, done, info, _ = self.env.step(action.item())
                step_count += 1
                self.agent.remember(observation, action, reward, observation_, done)
                self.agent.learn()
                score += reward
                observation = observation_
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            print('episode ', i, 'score %.1f' % score,
                    'average score %.1f' % avg_score)
            
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.agent.save_models()
            
            if (i+1) % 100 == 0:
                np.save("episode_rewards/cartpole_e_r", np.array(self.score_history))

        np.save("episode_rewards/cartpole_e_r", np.array(self.score_history))
