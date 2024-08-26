##############################################################################

from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv
from Examples.RL.RLExample import RLExample

##############################################################################

# Environment
env = CustomContinuousCartPoleEnv()
model_name = "cartpole_test"
input_dims = [4]
n_actions = 1
sigmoid = False
action_bound = 2

# Training
load_checkpoint = False
ngames = 2000
max_steps_per_episode = 1000 #TODO what if this is closer to the simulation time?
sigma_noise = 0.5

# Evaluation
dt = 0.02
total_simulation_time = 20
state_labels = ['Position (m)', 'Velocity (m/s)', 'Angle (rads)', 'Angular Velocity (rads/s)']
control_labels = ['Force (N)']

##############################################################################

CartPoleExample = RLExample(env=env, model_name=model_name, n_actions=n_actions, input_dims=input_dims, sigmoid=sigmoid, action_bound=action_bound)

CartPoleExample.train(load_checkpoint=load_checkpoint, ngames=ngames, max_steps_per_episode=max_steps_per_episode, sigma_noise=sigma_noise)

# CartPoleExample.evaluate(dt=dt, total_simulation_time=total_simulation_time, state_labels=state_labels, control_labels=control_labels)
# CartPoleExample.plot_rewards()

