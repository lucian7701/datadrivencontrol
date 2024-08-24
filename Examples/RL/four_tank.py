import os
import numpy as np
from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Models.rl_model_own_four_tank import CustomFourTankEnv
from Analysis.RL.episode_reward import plot_rewards


model_name = "four_tank_simga_1.5"
# model_name = "four_tank"

def train_four_tank():
    env = CustomFourTankEnv()

    ddpg_controller = DDPGTrainer(env, model_name=model_name, load_checkpoint=False, input_dims=[4], n_actions=2, ngames=2000, max_steps_per_episode=300, sigmoid=True, action_bound=100, sigma_noise=1.5)
    ddpg_controller.run()


def evaluate_four_tank():
    env = CustomFourTankEnv()
    ddpg_evaluator = DDPGEvaluator(env, model_name=model_name, input_dims=[4], n_actions=2, dt=3, total_simulation_time=240, sigmoid=True, action_bound=100)
   
    state_labels = ['Tank 1 (m)', 'Tank 2 (m)', 'Tank 3 (m)', 'Tank 4 (m)']
    control_labels = ['Valve 1 (m^3/s)', 'Valve 2 (m^3/s)']
    ddpg_evaluator.run_eval(state_labels=state_labels, control_labels=control_labels)


def plot_rewards_four_tank():
    rewards_file_path = os.path.join(os.path.dirname(__file__), f'episode_rewards/{model_name}_e_r.npy')

    if os.path.exists(rewards_file_path):
        saved_rewards = np.load(rewards_file_path)
        plot_rewards(saved_rewards)
    else: 
        print("File not found")


##############################################################################

# train_four_tank()


evaluate_four_tank()


# plot_rewards_four_tank()
