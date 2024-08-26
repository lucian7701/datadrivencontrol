# import os
# import numpy as np
# from Controllers.RL.DDPGTrainer import DDPGTrainer
# from Controllers.RL.DDPGEvaluator import DDPGEvaluator
# from Models.rl_model_own_di import CustomDoubleIntegratorEnv
# from Analysis.RL.episode_reward import plot_rewards


# def train_double_integrator():
#     env = CustomDoubleIntegratorEnv(target_position=7.0)

#     ddpg_controller = DDPGTrainer(env, model_name="double_integrator", load_checkpoint=False, input_dims=[1], n_actions=1, ngames=2000, max_steps_per_episode=300)
#     ddpg_controller.run()


# def evaluate_double_integrator():
#     env = CustomDoubleIntegratorEnv(target_position=7.0)
#     ddpg_evaluator = DDPGEvaluator(env, model_name="double_integrator", input_dims=[1], n_actions=1)
   
#     state_labels = ['Position (m)', 'Velocity (m/s)']
#     control_labels = ['Force (N)']
#     ddpg_evaluator.run_eval(state_labels=state_labels, control_labels=control_labels)


# def plot_rewards_double_integrator():
#     rewards_file_path = os.path.join(os.path.dirname(__file__), 'episode_rewards/double_integrator_e_r.npy')

#     if os.path.exists(rewards_file_path):
#         saved_rewards = np.load(rewards_file_path)
#         plot_rewards(saved_rewards)
#     else: 
#         print("File not found")


# ##############################################################################

# # train_double_integrator()


# evaluate_double_integrator()


# # plot_rewards_double_integrator()



##############################################################################

from Models.rl_model_own_di import CustomDoubleIntegratorEnv
from Examples.RL.RLExample import RLExample

##############################################################################

# Double integrator partial observation

# Environment
env_partial = CustomDoubleIntegratorEnv(target_position=7.0)
model_name_partial = "partial"
input_dims_partial = [1]
n_actions_partial = 1
sigmoid_partial = False
action_bound_partial = 2
chpt_dir_partial = "double_integrator"

# Training
load_checkpoint_partial = False
ngames_partial = 2000
max_steps_per_episode_partial = 300
sigma_noise_partial = 0.5

# Evaluation
dt_partial = 0.02
total_simulation_time_partial = 20
state_labels_partial = ['Position (m)', 'Velocity (m/s)']
control_labels_partial = ['Force (N)']

##############################################################################

DoubleIntegratorPartialExample = RLExample(env=env_partial, model_name=model_name_partial, n_actions=n_actions_partial, input_dims=input_dims_partial, sigmoid=sigmoid_partial, action_bound=action_bound_partial, chkpt_dir=chpt_dir_partial)

DoubleIntegratorPartialExample.train(load_checkpoint=load_checkpoint_partial, ngames=ngames_partial, max_steps_per_episode=max_steps_per_episode_partial, sigma_noise=sigma_noise_partial)

# DoubleIntegratorPartialExample.evaluate(dt=dt_partial, total_simulation_time=total_simulation_time_partial, state_labels=state_labels_partial, control_labels=control_labels_partial)

##############################################################################
##############################################################################

# Double integrator full observation

# Environment
env_full = CustomDoubleIntegratorEnv(target_position=7.0)
model_name_full = "full"
input_dims_full = [2]
n_actions_full = 1
sigmoid_full = False
action_bound_full = 2
chpt_dir_full = "double_integrator"

# Training
load_checkpoint_full = False
ngames_full = 2000
max_steps_per_episode_full = 300
sigma_noise_full = 0.5

# Evaluation
dt_full = 0.02
total_simulation_time_full = 20
state_labels_full = ['Position (m)', 'Velocity (m/s)']
control_labels_full = ['Force (N)']

##############################################################################

DoubleIntegratorFullExample = RLExample(env=env_full, model_name=model_name_full, n_actions=n_actions_full, input_dims=input_dims_full, sigmoid=sigmoid_full, action_bound=action_bound_full, chkpt_dir=chpt_dir_full)

DoubleIntegratorFullExample.train(load_checkpoint=load_checkpoint_full, ngames=ngames_full, max_steps_per_episode=max_steps_per_episode_full, sigma_noise=sigma_noise_full)

# DoubleIntegratorFullExample.evaluate(dt=dt_full, total_simulation_time=total_simulation_time_full, state_labels=state_labels_full, control_labels=control_labels_full)

##############################################################################


