import numpy as np
import matplotlib.pyplot as plt

def plot_average_errors_by_state(average_errors, std_errors, episode_number, state_labels=None):
    """
    Plots the average errors with standard deviation error bars for multiple states.

    :param average_errors: A 2D numpy array where each row corresponds to an episode and each column to a state.
    :param std_errors: A 2D numpy array with the same shape as average_errors containing standard deviations.
    :param episode_number: A list or array of episode numbers.
    :param state_labels: A list of strings representing the labels for each state. Defaults to None.
    """

    num_states = average_errors.shape[1]  # Number of states
    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})

    print(episode_number[:5])

    for state_index in range(num_states):
        # Extract the average and std for the current state across all episodes
        avg = average_errors[:, state_index]
        std = std_errors[:, state_index]

        # Clip the lower bound of the error bars at zero
        lower_bound = np.clip(avg - std, 0, None)
        upper_bound = avg + std

        # Calculate the asymmetric error bars
        error_bars = [avg - lower_bound, upper_bound - avg]

        # Generate a label for the state if state_labels are provided
        label = state_labels[state_index] if state_labels else f'State {state_index+1}'

        # Plot the error bar
        plt.errorbar(episode_number, avg, yerr=error_bars, label=label, capsize=5, marker='o', linestyle='--')

    # Adding plot details
    plt.xlabel("Episode number")
    plt.ylabel("Average total error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_total_average_errors(total_average_errors, total_std_errors, episode_number, state_labels=None):
    """
    Plots the total average errors with standard deviation error bars.

    :param total_average_errors: A 1D numpy array containing the total average errors for each episode.
    :param total_std_errors: A 1D numpy array containing the total standard deviations for each episode.
    :param episode_number: A list or array of episode numbers.
    """

    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})

    # Clip the lower bound of the error bars at zero
    lower_bound = np.clip(total_average_errors - total_std_errors, 0, None)
    upper_bound = total_average_errors + total_std_errors

    error_bars = [total_average_errors - lower_bound, upper_bound - total_average_errors]

    plt.errorbar(episode_number, total_average_errors, yerr=error_bars, capsize=5, marker='o', linestyle='--')

    # Adding plot details
    plt.xlabel("Episode number")
    plt.ylabel("Average total error")
    plt.grid(True)
    plt.show()




def plot_average_controls_by_input(average_controls, std_controls, episode_number, control_labels=None):
    """
    Plots the average controls with standard deviation error bars for multiple inputs.

    :param average_controls: A 2D numpy array where each row corresponds to an episode and each column to a control input.
    :param std_controls: A 2D numpy array with the same shape as average_controls containing standard deviations.
    :param episode_number: A list or array of episode numbers.
    """

    num_controls = average_controls.shape[1]  # Number of control inputs
    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})

    for control_index in range(num_controls):
        # Extract the average and std for the current control input across all episodes
        avg = average_controls[:, control_index]
        std = std_controls[:, control_index]

        label = control_labels[control_index] if control_labels else f'Control {control_index+1}'
        # Plot the error bar
        plt.errorbar(episode_number, avg, yerr=std, label=label, capsize=5, marker='o', linestyle='--')

    # Adding plot details
    plt.xlabel("Episode number")
    plt.ylabel("Total control effort")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_total_average_control(total_average_control, total_std_control, episode_number, control_labels=None):
    """
    Plots the total average control with standard deviation error bars.

    :param total_average_control: A 1D numpy array containing the total average control for each episode.
    :param total_std_control: A 1D numpy array containing the total standard deviation for each episode.
    :param episode_number: A list or array of episode numbers.
    """

    plt.figure(figsize=(7, 4))
    plt.rcParams.update({'font.size': 10})
    plt.errorbar(episode_number, total_average_control, yerr=total_std_control, capsize=5, marker='o', linestyle='--')

    # Adding plot details
    plt.xlabel("Episode number")
    plt.ylabel("Total control effort")
    plt.grid(True)
    plt.show()
