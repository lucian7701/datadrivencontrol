""" This will plot state, control and reference """


# Plotting function

import numpy as np
import matplotlib.pyplot as plt

def plot_results(states, controls, T, title, reference_trajectory=None, with_limit_lines=False):
    time = np.linspace(0, T, len(states))
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    
    plt.subplot(2, 1, 1)
    plt.plot(time, states[:,0], label='Position 1')

    if states.shape[1] > 1:
        plt.plot(time, states[:,1], label='Position 2')
    if states.shape[1] > 2:
        plt.plot(time, states[:,2], label='Position 3')
    if states.shape[1] > 3:
        plt.plot(time, states[:,3], label='Position 4')
        
    if reference_trajectory is not None:
        plt.plot(time, reference_trajectory, label='Reference Trajectory', linestyle='--')
    plt.ylabel('State')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time, controls, label='Control Input')
    plt.ylabel('Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()
    
    plt.show()

