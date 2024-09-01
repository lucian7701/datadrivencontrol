import numpy as np
import matplotlib.pyplot as plt

def plot_data_usage(data, num_controls=1, num_states=1):
    """data is number of steps in each episode. length is number of episodes"""

    data_points_per_episode = np.array(data) * (num_states + num_controls)

    cumulative_data_points = np.cumsum(data_points_per_episode)


    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})
    plt.plot(cumulative_data_points, label='Data Usage')

    plt.ylabel('Cumulative number of data points')
    plt.xlabel('Episode number')
    plt.grid(True)
    plt.show()
    