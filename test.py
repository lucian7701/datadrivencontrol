import numpy as np
from scipy import signal as scipysig


A = np.array([[0, 1],
              [0, 0]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

dt = 0.05

x0 = np.zeros(A.shape[0])

sys = scipysig.StateSpace(A, B, C, D).to_discrete(dt)

u = np.array([[1], [2], [3], [4], [5]])

t, y, x = scipysig.dlsim(sys, u, t = np.arange(len(u)) * dt , x0 = x0)


print(y)
print(x)
