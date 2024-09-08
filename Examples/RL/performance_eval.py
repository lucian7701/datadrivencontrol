import numpy as np
import os

data = np.load("performance/inverted_pendulum.npz")


# calculate mean total error
print("mean total error", data['errors'].mean())
# total error std
print("total error std", data['errors'].std())


# mean total control
print("mean total control", data['controls'].mean())
# total control std
print("total control std", data['controls'].std())


# by state

# mean total error by state
print("mean total error by state", data['errors_by_state'].mean(axis=0))
# total error by state std
print("total error by state std", data['errors_by_state'].std(axis=0))

# mean total control by input
print("mean total control by input", data['controls_by_input'].mean(axis=0))
# total control by input std
print("total control by input std", data['controls_by_input'].std(axis=0))




print("tanks", data['errors_by_state'])