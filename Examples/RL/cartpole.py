##############################################################################

from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv
from Examples.RL.RLExample import RLExample

##############################################################################

# Environment
env = CustomContinuousCartPoleEnv()
model_name = "cartpole"
input_dims = [4]
n_actions = 1
sigmoid = False
action_bound = 2
problem = "cartpole"
filename = "inverted_pendulum"

# Training
load_checkpoint = False
ngames = 2000
max_steps_per_episode = 1000 
sigma_noise = 0.5
gamma = 0.99
theta_noise = 0.15

# Evaluation
dt = 0.02
total_simulation_time = 10
state_labels = ['Position (m)', 'Velocity (m/s)', 'Angle (rads)', 'Angular Velocity (rads/s)']
control_labels = ['Force (N)']
ref_labels = "None"

##############################################################################

CartPoleExample = RLExample(env=env, model_name=model_name, n_actions=n_actions, input_dims=input_dims, sigmoid=sigmoid, action_bound=action_bound, problem=problem, dt=dt)

# CartPoleExample.train_with_eval(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise, gamma=gamma, theta_noise=theta_noise, total_simulation_time=total_simulation_time)

CartPoleExample.evaluate(total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels, filename=filename)
# CartPoleExample.plot_rewards()
# CartPoleExample.plot_average_stats(state_labels=state_labels, control_labels=control_labels)
# CartPoleExample.plot_data_usage()
