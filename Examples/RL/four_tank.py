##############################################################################

from Models.rl_model_own_four_tank import CustomFourTankEnv
from Examples.RL.RLExample import RLExample
import numpy as np

##############################################################################


##############################################################################
# Generic variables

# Environment
env = CustomFourTankEnv()
n_actions = 2
input_dims = [4]
sigmoid = True
action_bound = 100
target_positions = np.array([14, 14, 14.2, 21.3])
problem = "four_tank"

# Training
load_checkpoint = False
ngames = 2000
max_steps_per_episode = 100
sigma_noise = 5
theta_noise = 0.05
gamma = 0.99

# Evaluation
dt = 3
total_simulation_time = 240
state_labels = ['Tank 1 (m)', 'Tank 2 (m)', 'Tank 3 (m)', 'Tank 4 (m)']
control_labels = ['Pump 1 (m^3/s)', 'Pump 2 (m^3/s)']
ref_labels = ['Tank 1 ref (m)', 'Tank 2 ref (m)', 'Tank 3 ref (m)', 'Tank 4 ref (m)']






# ##############################################################################
# ##############################################################################

# # Environment 
# model_name = "four_tank_theta_0.05_simga_5"

# # Training


# # Evaluation


# ##############################################################################

# FourTankExample = RLExample(env=env, model_name=model_name, n_actions=n_actions, input_dims=input_dims, sigmoid=sigmoid, action_bound=action_bound, problem=problem, dt=dt)

# FourTankExample.train_with_eval(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise, gamma=gamma, theta_noise=theta_noise, total_simulation_time=total_simulation_time)

# FourTankExample.evaluate(total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels)
# # FourTankExample.plot_rewards()
# # FourTankExample.plot_average_stats(state_labels=state_labels, control_labels=control_labels)
# # FourTankExample.plot_data_usage()


##############################################################################
##############################################################################
# Four tank with theta = 0.15 and sigma = 5 but with max steps 100

# Environment 
model_name_100 = "theta_0.15_simga_1.5_max_steps_100"

# Training
max_steps_per_episode = 100
theta_noise = 0.15
sigma_noise = 1.5
# Evaluation


##############################################################################

FourTankExample = RLExample(env=env, model_name=model_name_100, n_actions=n_actions, input_dims=input_dims, sigmoid=sigmoid, action_bound=action_bound, problem=problem, dt=dt)

# FourTankExample.train_with_eval(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise, gamma=gamma, theta_noise=theta_noise, total_simulation_time=total_simulation_time)

# FourTankExample.evaluate(total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels, ref_labels=ref_labels)
# FourTankExample.plot_rewards()
# FourTankExample.plot_average_stats(state_labels=state_labels, control_labels=control_labels)
FourTankExample.plot_data_usage()

