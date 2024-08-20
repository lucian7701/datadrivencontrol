import numpy as np
import os
from Analysis.RL.episode_reward import plot_rewards


rewards_file_path = os.path.join(os.path.dirname(__file__), 'episode_rewards/cartpole_e_r.npy')

if os.path.exists(rewards_file_path):
    saved_rewards = np.load(rewards_file_path)
    print(saved_rewards)


else:
    print("File not found")


plot_rewards(saved_rewards)
