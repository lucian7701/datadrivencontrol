import os
import numpy as np
from Analysis.Analyser import Analyser


# filename = 'double_integrator'

# file_path = os.path.join(os.getcwd(), f'performance/{filename}.npz')

# data = np.load(file_path)

# states = data['arr_0']
# states = states[:-1]
# control = data['arr_1']

# first_prediction_horizon_mean = data['arr_2']
# first_prediction_horizon_std = data['arr_3']
# first_prediction_horizon_mean = first_prediction_horizon_mean[:-1]
# first_prediction_horizon_std = first_prediction_horizon_std[:-1]

# print(first_prediction_horizon_mean, first_prediction_horizon_std)

# y_ref = np.array([7.0, 0])

# analyser = Analyser(states=states, controls=control, reference_trajectory=y_ref.reshape(1,-1))

# analyser.plot_state_control(0.1, ['Position (m)', 'Velocity (m/s)'], ['Force (N)'], ref_labels=['Position ref (m)', 'Velocity ref (m/s)'], first_prediction_horizon_mean=first_prediction_horizon_mean, first_prediction_horizon_std=first_prediction_horizon_std)
# print(analyser.total_absolute_error_by_state())

#############################################################################

# data = np.load("performance/four_tank_total.npz")

# # calculate mean total error
# print("mean total error", data['errors'].mean())
# # total error std
# print("total error std", data['errors'].std())


# # mean total control
# print("mean total control", data['controls'].mean())
# # total control std
# print("total control std", data['controls'].std())


# # by state

# # mean total error by state
# print("mean total error by state", data['errors_by_state'].mean(axis=0))
# # total error by state std
# print("total error by state std", data['errors_by_state'].std(axis=0))

# # mean total control by input
# print("mean total control by input", data['controls_by_input'].mean(axis=0))
# # total control by input std
# print("total control by input std", data['controls_by_input'].std(axis=0))


# print("errors by state", data['errors_by_state'])


#############################################################################

# filename = 'four_tank'

# file_path = os.path.join(os.getcwd(), f'performance/{filename}.npz')

# data = np.load(file_path)

# states = data['arr_0']
# states = states[:-1]
# control = data['arr_1']

# first_prediction_horizon_mean = data['arr_2']
# first_prediction_horizon_std = data['arr_3']
# first_prediction_horizon_mean = first_prediction_horizon_mean[:-1]
# first_prediction_horizon_std = first_prediction_horizon_std[:-1]


# y_ref = np.array([14, 14, 14.2, 21.3])

# analyser = Analyser(states=states, controls=control, reference_trajectory=y_ref.reshape(1,-1))

# analyser.plot_state_control(0.1, ['Tank 1 (m)', 'Tank 2 (m)', 'Tank 3 (m)', 'Tank 4 (m)'], ['Pump 1 (m^3/s)', 'Pump 2 (m^3/s)'], ref_labels=['Tank 1 ref (m)', 'Tank 2 ref (m)', 'Tank 3 ref (m)', 'Tank 4 ref (m)'], first_prediction_horizon_mean=first_prediction_horizon_mean, first_prediction_horizon_std=first_prediction_horizon_std)

# print(analyser.total_absolute_error_by_state())


#############################################################################

# data = np.load("performance/inverted_pendulum_total.npz")

# # calculate mean total error
# print("mean total error", data['errors'].mean())
# # total error std
# print("total error std", data['errors'].std())


# # mean total control
# print("mean total control", data['controls'].mean())
# # total control std
# print("total control std", data['controls'].std())


# # by state

# # mean total error by state
# print("mean total error by state", data['errors_by_state'].mean(axis=0))
# # total error by state std
# print("total error by state std", data['errors_by_state'].std(axis=0))

# # mean total control by input
# print("mean total control by input", data['controls_by_input'].mean(axis=0))
# # total control by input std
# print("total control by input std", data['controls_by_input'].std(axis=0))


# print("errors by state", data['errors_by_state'])

#############################################################################

filename = 'inverted_pendulum'

file_path = os.path.join(os.getcwd(), f'performance/{filename}.npz')

data = np.load(file_path)

states = data['arr_0']
states = states[:-1]
control = data['arr_1']

first_prediction_horizon_mean = data['arr_2']
first_prediction_horizon_std = data['arr_3']
first_prediction_horizon_mean = first_prediction_horizon_mean[:-1]
first_prediction_horizon_std = first_prediction_horizon_std[:-1]


y_ref = np.array([0, 0, 0, 0])

analyser = Analyser(states=states, controls=control, reference_trajectory=y_ref.reshape(1,-1))

state_labels = ['Position (m)', 'Velocity (m/s)', 'Angle (rads)', 'Angular velocity (rads/s)']
control_labels = ['Force (N)']

analyser.plot_state_control(0.02, state_labels, control_labels, ref_labels="None", first_prediction_horizon_mean=first_prediction_horizon_mean, first_prediction_horizon_std=first_prediction_horizon_std, plot_first_prediction_horizon=True)

print(analyser.total_absolute_error_by_state())