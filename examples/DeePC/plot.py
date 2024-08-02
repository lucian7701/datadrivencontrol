import numpy as np
import matplotlib.pyplot as plt
import os

# Plotting function

def plot_results(states, controls, T, title, with_limit_lines=False, with_video=False, video_dir=None, figure_dir=None, figure_name=None, linear=False):


    time = np.linspace(0, T, len(states))
    plt.figure(figsize=(12, 8))
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    if not linear:
        plt.plot(time, states[:, 0], label='Cart Position')
        plt.plot(time, states[:, 2], label='Pole Angle')
        plt.ylabel('State')

    else:
        plt.plot(time, states[:, 0], label='Position')
        plt.plot(time, states[:, 1], label='Velocity')
        plt.ylabel('State')

    if with_limit_lines:
        plt.axhline(9*np.pi/180, color='r', linestyle='--', label='9 degrees limit')
        plt.axhline(-9*np.pi/180, color='r', linestyle='--')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(time[:-1], controls, label='Control Input')
    plt.ylabel('Control Input')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()

    if with_video:
        print(f"Video saved in folder {video_dir}")

    if figure_dir:
        # Save the figure in the specified folder
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        file_path = os.path.join(figure_dir, figure_name + '.png')
        plt.savefig(file_path)
        print(f"Figure saved in folder {figure_dir}")
