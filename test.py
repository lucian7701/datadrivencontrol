import numpy as np
from scipy import signal as scipysig


y_min = np.array([0, 0.5])
y_max = np.array([0.2, 1.5])


print(np.all(y_min < y_max))