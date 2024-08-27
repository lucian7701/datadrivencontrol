import numpy as np
from Controllers.RL.ddpg_torch import Agent
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
import json


class DDPGTrainer:

    def __init__(self, env, model_name, input_dims=[4], n_actions=1, load_checkpoint=False, ngames=2500, max_steps_per_episode=1000, action_bound=2, sigmoid=False, sigma_noise=0.5, problem='tmp/ddpg', dt=0.1, gamma=0.99, theta_noise=0.15):
        """
        env: gym environment
        """
        self.env = env
        self.model_name = model_name
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.n_games = ngames
        self.max_steps_per_episode = max_steps_per_episode
        self.action_bound = action_bound
        self.sigmoid = sigmoid
        self.problem = problem
        self.dt = dt
        self.chkpt_dir = f"trained_agents/{problem}"
        self.agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=input_dims, tau=0.005,
                    batch_size=64, fc1_dims=400, fc2_dims=300, 
                    n_actions=n_actions, model_name=model_name, gamma=gamma, action_bound=action_bound, sigmoid=sigmoid, sigma_noise=sigma_noise, chkpt_dir=self.chkpt_dir, dt=dt, theta_noise=theta_noise)
        
        self.best_score = self.env.reward_range[0]
        self.score_history = []
        self.average_stats = []

        if load_checkpoint:
            self.agent.load_models()

    
    def run(self, eval: bool = False, total_simulation_time = 20):

        overall_data_count = 0

        for i in range(self.n_games):
            observation, _ = self.env.reset()
            done = False
            score = 0
            self.agent.noise.reset()
            step_count = 0
            while not done and step_count < self.max_steps_per_episode:
                action = self.agent.choose_action(observation)
                
                if action.shape[0] == 1:
                    action = action.item()

                observation_, reward, done, info, _ = self.env.step(action)
                step_count += 1
                self.agent.remember(observation, action, reward, observation_, done)
                self.agent.learn()
                score += reward
                observation = observation_
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            overall_data_count += step_count

            print('episode ', i, 'score %.1f' % score,
                    'average score %.1f' % avg_score)
            
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.agent.save_models()
            
            if (i+1) % 10 == 0:
                
                with open(f"data_usage/{self.problem}_{self.model_name}.txt", "w") as f:
                    f.write(f"{overall_data_count}")
                
                np.save(f"episode_rewards/{self.problem}_{self.model_name}_e_r", np.array(self.score_history))

                if eval:

                    errors_by_state_list = []
                    controls_by_input_list = []
                    total_errors_list = []
                    total_controls_list = []

                    for _ in range(10):

                        ddpg_evaluator = DDPGEvaluator(self.env, model_name=self.model_name, input_dims=self.input_dims, n_actions=self.n_actions, dt=self.dt, total_simulation_time=total_simulation_time, sigmoid=self.sigmoid, action_bound=self.action_bound, problem=self.problem)
                        total_absolute_error_by_state, total_absolute_control_by_input = ddpg_evaluator.run_eval(plot=False)

                        errors_by_state_list.append(total_absolute_error_by_state)
                        controls_by_input_list.append(total_absolute_control_by_input)

                        total_errors_list.append(np.sum(total_absolute_error_by_state))
                        total_controls_list.append(np.sum(total_absolute_control_by_input))


                    errors_by_state_array = np.array(errors_by_state_list)
                    controls_by_input_array = np.array(controls_by_input_list)
                    total_errors_array = np.array(total_errors_list)
                    total_controls_array = np.array(total_controls_list)


                    average_error_by_state = np.mean(errors_by_state_array, axis=0)
                    std_dev_error_by_state = np.std(errors_by_state_array, axis=0)
                    total_average_error = np.mean(total_errors_array)
                    total_std_dev_error = np.std(total_errors_array)


                    average_control_by_input = np.mean(controls_by_input_array, axis=0)
                    std_dev_control_by_input = np.std(controls_by_input_array, axis=0)
                    total_average_control = np.mean(total_controls_array)
                    total_std_dev_control = np.std(total_controls_array)

                    combined_error_stats = np.concatenate((average_error_by_state, std_dev_error_by_state))
                    combined_control_stats = np.concatenate((average_control_by_input, std_dev_control_by_input))
                    combined_total_stats = np.array([total_average_error, total_std_dev_error, total_average_control, total_std_dev_control])

                    combined_stats_row = np.concatenate((combined_error_stats, combined_control_stats, combined_total_stats))

                    # Append the episode number to the row
                    episode_number = np.array([i + 1])  # episode number is (i + 1)

                    # this has the format: [average_error_by_state, std_dev_error_by_state, average_control_by_input, std_dev_control_by_input, total_average_error, total_std_dev_error, total_average_control, total_std_dev_control, episode_number]
                    combined_stats_row = np.concatenate((combined_stats_row, episode_number))

                    self.average_stats.append(combined_stats_row)

                    np.save(f"evaluation_stats/{self.problem}_{self.model_name}_stats", np.array(self.average_stats))

                    
        np.save(f"episode_rewards/{self.problem}_{self.model_name}_e_r", np.array(self.score_history))
