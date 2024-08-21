import os
import numpy as np
from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Models.rl_model_own_cartpole import CustomContinuousCartPoleEnv
from Analysis.RL.episode_reward import plot_rewards


def train_cartpole():
    env = CustomContinuousCartPoleEnv()

    ddpg_controller = DDPGTrainer(env, model_name="cartpole2", load_checkpoint=False)
    ddpg_controller.run()


def evaluate_cartpole():
    env = CustomContinuousCartPoleEnv()
    ddpg_evaluator = DDPGEvaluator(env, model_name="cartpole")
   
    state_labels = ['Position (m)', 'Velocity (m/s)', 'Angle (rads)', 'Angular Velocity (rads/s)']
    control_labels = ['Force (N)']
    ddpg_evaluator.run_eval(state_labels=state_labels, control_labels=control_labels)


def plot_rewards_cartpole():
    rewards_file_path = os.path.join(os.path.dirname(__file__), 'episode_rewards/cartpole_e_r.npy')

    if os.path.exists(rewards_file_path):
        saved_rewards = np.load(rewards_file_path)
        plot_rewards(saved_rewards)


##############################################################################

# train_cartpole()


evaluate_cartpole()


# plot_rewards_cartpole()
