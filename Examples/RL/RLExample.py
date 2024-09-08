import os
import numpy as np
from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Analysis.RL.episode_reward import plot_rewards
from Analysis.RL.errors_controls import plot_average_errors_by_state, plot_average_controls_by_input, plot_total_average_errors, plot_total_average_control
from Analysis.RL.data_usage import plot_data_usage

class RLExample:
    def __init__(self, env, model_name, n_actions, input_dims, problem, sigmoid=False, action_bound=2, dt=0.1):
        self.env = env
        self.model_name = model_name
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.sigmoid = sigmoid
        self.action_bound = action_bound
        self.problem = problem
        
        self.dt = dt
       
        
    def train(self, load_checkpoint=False, ngames=2000, max_steps_per_episode=300, sigma_noise=1.5, gamma=0.99, theta_noise=0.15):
        ddpg_controller = DDPGTrainer(self.env, model_name=self.model_name, 
                                    load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise,
                                    sigmoid=self.sigmoid, action_bound=self.action_bound, input_dims=self.input_dims, n_actions=self.n_actions, problem=self.problem, dt=self.dt, gamma=gamma, theta_noise=theta_noise)
        ddpg_controller.run()


    def train_with_eval(self, load_checkpoint=False, ngames=2000, max_steps_per_episode=300, sigma_noise=1.5, gamma=0.99, theta_noise=0.15, total_simulation_time=20):
        ddpg_controller = DDPGTrainer(self.env, model_name=self.model_name, 
                                    load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise,
                                    sigmoid=self.sigmoid, action_bound=self.action_bound, input_dims=self.input_dims, n_actions=self.n_actions, problem=self.problem, dt=self.dt, gamma=gamma, theta_noise=theta_noise)
        ddpg_controller.run(eval=True, total_simulation_time=total_simulation_time)


    def evaluate(self, total_simulation_time=20, state_labels=None, control_labels=None, ref_labels=None, filename=None):
        ddpg_evaluator = DDPGEvaluator(self.env, model_name=self.model_name, input_dims=self.input_dims, n_actions=self.n_actions, dt=self.dt, total_simulation_time=total_simulation_time, sigmoid=self.sigmoid, action_bound=self.action_bound, problem=self.problem)
        
        ddpg_evaluator.run_eval(state_labels, control_labels, ref_labels, filename=filename)

    def plot_rewards(self):
        rewards_file_path = os.path.join(os.path.dirname(__file__), f'episode_rewards/{self.problem}_{self.model_name}_e_r.npy')

        if os.path.exists(rewards_file_path):
            saved_rewards = np.load(rewards_file_path)
            plot_rewards(saved_rewards)
        else: 
            print("File not found")

    def plot_average_stats(self, state_labels=None, control_labels=None):
        evaluation_stats_file_path = os.path.join(os.path.dirname(__file__), f"evaluation_stats/{self.problem}_{self.model_name}_stats.npy")
        
        if os.path.exists(evaluation_stats_file_path):
            evaluation_stats = np.load(evaluation_stats_file_path)

            average_errors_by_state = evaluation_stats[:, :self.input_dims[0]]
            std_errors_by_state = evaluation_stats[:, self.input_dims[0]:2*self.input_dims[0]]
            average_controls_by_input = evaluation_stats[:, 2*self.input_dims[0]:2*self.input_dims[0]+self.n_actions]
            std_controls_by_input = evaluation_stats[:, 2*self.input_dims[0]+self.n_actions:2*self.input_dims[0]+2*self.n_actions]
            
            total_average_errors = evaluation_stats[:, 2*self.input_dims[0]+2*self.n_actions]
            total_std_errors = evaluation_stats[:, 2*self.input_dims[0]+2*self.n_actions+1]
            total_average_controls = evaluation_stats[:, 2*self.input_dims[0]+2*self.n_actions+2]
            total_std_controls = evaluation_stats[:, 2*self.input_dims[0]+2*self.n_actions+3]
            
            episode_number = evaluation_stats[:, -1]

            plot_average_errors_by_state(average_errors_by_state, std_errors_by_state, episode_number, state_labels)
            plot_average_controls_by_input(average_controls_by_input, std_controls_by_input, episode_number, control_labels)

            plot_total_average_errors(total_average_errors, total_std_errors, episode_number, state_labels)
            plot_total_average_control(total_average_controls, total_std_controls, episode_number, control_labels)

        else:
            print("File not found")

    def plot_data_usage(self):
        data_usage_file_path = os.path.join(os.path.dirname(__file__), f'data_usage/{self.problem}_{self.model_name}.npy')

        if os.path.exists(data_usage_file_path):
            data_usage = np.load(data_usage_file_path)
            plot_data_usage(data_usage, num_controls=self.n_actions, num_states=self.input_dims[0])

        else:
            print("File not found")
        