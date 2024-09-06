##############################################################################

from Models.rl_model_di_partial import CustomDoubleIntegratorEnv
from Models.rl_model_di_full import CustomDoubleIntegratorFullEnv
from Examples.RL.RLExample import RLExample
import numpy as np

##############################################################################


##############################################################################
# Generic variables

# Environment
target_positions = np.array([7, 0])
n_actions = 1
sigmoid = False
problem = "double_integrator"
action_bound = 2

# Training
load_checkpoint = False
ngames = 2000
max_steps_per_episode = 200
sigma_noise = 0.3
theta_noise = 0.00001
gamma = 0.99

# Evaluation
dt = 0.1
total_simulation_time = 20
state_labels = ['Position (m)', 'Velocity (m/s)']
control_labels = ['Force (N)']
ref_labels = ['Position ref (m)', 'Velocity ref (m/s)']

##############################################################################



# ##############################################################################
# ##############################################################################
# # Double integrator partial observation

# # Environment
# env_partial = CustomDoubleIntegratorEnv(target_position=7, dt=dt)
# model_name_partial = "partial1"
# input_dims_partial = [1]

# # Training

# # Evaluation

# ##############################################################################

# # DoubleIntegratorPartialExample = RLExample(env=env_partial, model_name=model_name_partial, n_actions=n_actions, input_dims=input_dims_partial, sigmoid=sigmoid, action_bound=action_bound, chkpt_dir=chkpt_dir)

# # DoubleIntegratorPartialExample.train(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise)

# # DoubleIntegratorPartialExample.evaluate(dt=dt, total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels)

# ##############################################################################
# ##############################################################################



# ##############################################################################
# ##############################################################################
# # Double integrator full observation with negative QR reward

# # Environment
# Q = np.array([[10, 0], [0, 1]])
# R = 0.01
# env_full_qr_reward = CustomDoubleIntegratorFullEnv(target_positions=target_positions, dt=dt, Q=Q, R=R)
# model_name_qr_reward = "full_QR_reward"
# input_dims_qr_reward = [2]

# # Training
# sigma_noise_qr_reward = 0.3
# theta_noise_qr_reward = 0.00001

# # Evaluation

# ##############################################################################

# # DoubleIntegratorFullExample = RLExample(env=env_full_qr_reward, model_name=model_name_qr_reward, n_actions=n_actions, input_dims=input_dims_qr_reward, sigmoid=sigmoid, action_bound=action_bound, problem=problem, dt=dt)

# # DoubleIntegratorFullExample.train(load_checkpoint=True, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise_qr_reward, gamma=gamma, theta_noise=theta_noise_qr_reward)

# # DoubleIntegratorFullExample.evaluate(total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels)
# # DoubleIntegratorFullExample.plot_rewards()

# ##############################################################################
# ##############################################################################





##############################################################################
##############################################################################
# Double integrator full observation with exponential reward (final)

# Environment
Q = np.array([[10, 0], [0, 1]])
R = 0.1
env_full_exp = CustomDoubleIntegratorFullEnv(target_positions=target_positions, dt=dt, Q=Q, R=R)
model_name_exp = "exponential_reward"
input_dims_exp= [2]

# Training
sigma_noise_exp = 0.5
theta_noise_exp = 0.15

# Evaluation

##############################################################################

DoubleIntegratorFullExample = RLExample(env=env_full_exp, model_name=model_name_exp, n_actions=n_actions, input_dims=input_dims_exp, sigmoid=sigmoid, action_bound=action_bound, problem=problem, dt=dt)

# DoubleIntegratorFullExample.train_with_eval(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise_exp, gamma=gamma, theta_noise=theta_noise_exp, total_simulation_time=total_simulation_time)

DoubleIntegratorFullExample.evaluate(total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels)
# DoubleIntegratorFullExample.plot_rewards()
# DoubleIntegratorFullExample.plot_average_stats(state_labels=state_labels, control_labels=control_labels)
# DoubleIntegratorFullExample.plot_data_usage()

##############################################################################
##############################################################################

