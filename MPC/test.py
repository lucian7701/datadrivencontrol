import numpy as np
from scipy import sparse


# iu = np.ones(3)
#
# iu = np.array([2,3,4,5,4])
#
# print(sparse.diags(iu).toarray())

Ad = np.array([[1, 1],
               [0, 1]])

Bd = np.array([[0.5],
               [1]])


xmin = np.array([-2, -2])

lineq_x = np.kron(np.ones(3 + 1), xmin)

Np = 3
nx = 2
x0 = np.array([1, 1])


leq_dyn = np.hstack([-x0, np.zeros(Np * nx)])

print(leq_dyn)
# print(lineq_x)


Ax = sparse.kron(sparse.eye(2 + 1), -sparse.eye(2)) + sparse.kron(sparse.eye(2 + 1, k=-1), Ad)

iBu = sparse.vstack([sparse.csc_matrix((1, 2)),
                             sparse.eye(2)])

Bu = sparse.kron(iBu, Bd)

Aeq_dyn = sparse.hstack([Ax, Bu])

print(Aeq_dyn.toarray())