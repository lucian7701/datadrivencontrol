import os
import numpy as np
from Controllers.RL.DDPGTrainer import DDPGTrainer
from Controllers.RL.DDPGEvaluator import DDPGEvaluator
from Models.rl_model_own_di_full_state import CustomDoubleIntegratorFullEnv
from Analysis.RL.episode_reward import plot_rewards


def train_test_di_full():
    env = CustomDoubleIntegratorFullEnv(target_position=7.0)

    ddpg_controller = DDPGTrainer(env, model_name="test_di_full", load_checkpoint=False, input_dims=[2], n_actions=1, ngames=2000, max_steps_per_episode=300)
    ddpg_controller.run()


def evaluate_test_di_full():
    env = CustomDoubleIntegratorFullEnv(target_position=7.0)
    ddpg_evaluator = DDPGEvaluator(env, model_name="test_di_full", input_dims=[2], n_actions=1)
   
    state_labels = ['Position (m)', 'Velocity (m/s)']
    control_labels = ['Force (N)']
    ddpg_evaluator.run_eval(state_labels=state_labels, control_labels=control_labels)


def plot_rewards_test_di_full():
    rewards_file_path = os.path.join(os.path.dirname(__file__), 'episode_rewards/test_di_full_e_r.npy')

    if os.path.exists(rewards_file_path):
        saved_rewards = np.load(rewards_file_path)
        plot_rewards(saved_rewards)
    else: 
        print("File not found")


##############################################################################

# train_test_di_full()


evaluate_test_di_full()


# plot_rewards_test_di_full()
