import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(rewards):
    """Plot of rewards vs episodes"""
    
    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})
    plt.plot(rewards, label='Episode Reward')
    
    average_reward = []
    for idx in range(len(rewards)):
        avg_list = np.empty(shape=(1,), dtype=int)
        if idx < 100:
            avg_list = rewards[:idx+1]
        else:
            avg_list = rewards[idx-100:idx+1]
        average_reward.append(np.average(avg_list))
    
    plt.plot(average_reward, label='Average Reward')


    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    plt.legend()
    plt.grid()
    plt.show()

