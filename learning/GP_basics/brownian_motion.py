import numpy as np
import matplotlib.pyplot as plt


def brownian_motion():
    # Brownian motion

    # simulate 5 paths of brownian motion
    total_time = 1
    nb_steps = 75
    delta_t = total_time / nb_steps
    nb_processes = 5
    mean = 0.
    stdev = np.sqrt(delta_t)

    # Simulate brownian motions in 1D space
    distances = np.cumsum(np.random.normal(mean, stdev, (nb_processes, nb_steps)), axis=1)

    # Make the plots
    plt.figure(figsize=(6, 4))
    t = np.arange(0, total_time, delta_t)
    for i in range(nb_processes):
        plt.plot(t, distances[i, :])
    plt.title((
        'Brownian motion process\n '
        'Position over time for 5 independent realizations'))
    plt.xlabel('$t$ (time)', fontsize=13)
    plt.ylabel('$d$ (position)', fontsize=13)
    plt.xlim([-0, 1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    brownian_motion()