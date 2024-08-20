""" This will plot state, control and reference """


# Plotting function

import numpy as np
import matplotlib.pyplot as plt

def plot_results(states, controls, total_simulation_time, title, reference_trajectory=None, T_ini=0):
    
    
    time = np.linspace(0, total_simulation_time, len(states))
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)
    
    plt.subplot(2, 1, 1)

    for i in range(states.shape[1]):
        plt.plot(time, states[:,i], label='State {}'.format(i+1))

    if reference_trajectory is not None:
        for i in range(reference_trajectory.shape[1]):
            plt.plot(time[T_ini:], reference_trajectory[:,i], label='Reference Trajectory {}'.format(i+1), linestyle='--')

    
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
