##############################################################################

from Models.rl_model_own_four_tank import CustomFourTankEnv
from Examples.RL.RLExample import RLExample

##############################################################################

# Environment 
env = CustomFourTankEnv()
model_name = "four_tank_simga_1.5_100_steps"
input_dims = [4]
n_actions = 2
sigmoid = True
action_bound = 100

# Training
load_checkpoint = False
ngames = 2000
max_steps_per_episode = 100
sigma_noise = 1.5

# Evaluation
dt = 3
total_simulation_time = 240
state_labels = ['Tank 1 (m)', 'Tank 2 (m)', 'Tank 3 (m)', 'Tank 4 (m)']
control_labels = ['Valve 1 (m^3/s)', 'Valve 2 (m^3/s)']


##############################################################################

FourTankExample = RLExample(env=env, model_name=model_name, n_actions=n_actions, input_dims=input_dims, sigmoid=sigmoid, action_bound=action_bound)

# FourTankExample.train(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise)

FourTankExample.evaluate(dt=dt, total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels)
# FourTankExample.plot_rewards()
