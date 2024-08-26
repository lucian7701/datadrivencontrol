import os
import numpy as np
from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Analysis.RL.episode_reward import plot_rewards


class RLExample:
    def __init__(self, env, model_name, n_actions, input_dims, chkpt_dir, sigmoid=False, action_bound=2):
        self.env = env
        self.model_name = model_name
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.sigmoid = sigmoid
        self.action_bound = action_bound
        self.chkpt_dir = f"trained_agents/{chkpt_dir}"
       
        
    def train(self, load_checkpoint=False, ngames=2000, max_steps_per_episode=300, sigma_noise=1.5):
        ddpg_controller = DDPGTrainer(self.env, model_name=self.model_name, 
                                    load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise,
                                    sigmoid=self.sigmoid, action_bound=self.action_bound, input_dims=self.input_dims, n_actions=self.n_actions, chkpt_dir=self.chkpt_dir)
        ddpg_controller.run()

    def evaluate(self, dt=0.02, total_simulation_time=20, state_labels=None, control_labels=None):
        ddpg_evaluator = DDPGEvaluator(self.env, model_name=self.model_name, input_dims=self.input_dims, n_actions=self.n_actions, dt=dt, total_simulation_time=total_simulation_time, sigmoid=self.sigmoid, action_bound=self.action_bound, chkpt_dir=self.chkpt_dir)
        
        ddpg_evaluator.run_eval(state_labels, control_labels)

    def plot_rewards(self):
        rewards_file_path = os.path.join(os.path.dirname(__file__), f'episode_rewards/{self.chkpt_dir}_{self.model_name}_e_r.npy')

        if os.path.exists(rewards_file_path):
            saved_rewards = np.load(rewards_file_path)
            plot_rewards(saved_rewards)
        else: 
            print("File not found")
