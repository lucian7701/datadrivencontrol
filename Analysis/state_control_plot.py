# Plotting function

import numpy as np
import matplotlib.pyplot as plt

def plot_state_control_util(states, controls, dt, state_labels, control_labels, reference_trajectory=None, T_ini=0, ref_labels=None):
    
    time = np.linspace(0, len(states)*dt, len(states))
    
    plt.figure(figsize=(7, 6))
    plt.rcParams.update({'font.size': 10})
    

    ##########################################################################
    # States plot
    if state_labels is None: 
        state_labels = ['State {}'.format(i+1) for i in range(states.shape[1])]

    plt.subplot(2, 1, 1)
    for i in range(states.shape[1]):
        plt.plot(time, states[:,i], label=state_labels[i])


    if ref_labels is None:
        ref_labels = ['Reference {}'.format(i+1) for i in range(reference_trajectory.shape[1])]
    if reference_trajectory is not None:
        plt.plot(time[T_ini:], [reference_trajectory[0]] * len(time[T_ini:]), label=ref_labels, linestyle='--')


    plt.ylabel('State')
    plt.legend()
    plt.grid()


    ##########################################################################
    # Control input plot
    if control_labels is None: 
        if controls.shape[1] == 1:
            control_labels = ['Control Input']
        else:
            control_labels = ['Control Input {}'.format(i+1) for i in range(controls.shape[1])]
    
    plt.subplot(2, 1, 2)
    for i in range(controls.shape[1]):
        plt.plot(time, controls[:,i], label=control_labels[i])

    plt.ylabel('Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()
    
    plt.show()
