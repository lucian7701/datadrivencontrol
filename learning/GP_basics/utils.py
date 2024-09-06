import numpy as np
import scipy as sp
from kernels import ExponentiatedQuadraticKernel

default_squared_exponential = ExponentiatedQuadraticKernel()
report_squared_exponential = ExponentiatedQuadraticKernel(sigma_f=0.5, sigma_n=0.1)


def GP(X1, y1, X2, kernel_func):
    """
    Calculate the posterior mean and covariance matrix for y2
    based on the corresponding input X2, the observations (y1, X1),
    and the prior kernel function. This all assumes mean prior mu=0

    ie, make n2 new samples based on n1 previously observed points.
    will use posterior distribution y2 given y1.
    y1 and y2 are jointly gaussian.
    get the conditional distribution mu 2|1 and Sigma 2|1
    can then predict y2 corresponding to input samples X2.
    """

    # Kernel of the observations
    Sigma11 = kernel_func(X1, X1)
    # Kernel of observations vs to-predict
    Sigma12 = kernel_func(X1, X2)
    # Solve. This basically finds (Sigma11^-1 * Sigma12) transposed // assume_a='pos' assumes its positive definite.
    # At this point sparse GP would be used. Note that Cholesky is done behind the scenes here.
    solved = sp.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
    # Compute posterior mean - mu 2|1
    mu2 = solved @ y1
    # Compute posterior covariance - Sigma 2|1
    Sigma22 = kernel_func(X2, X2)
    Sigma2 = Sigma22 - (solved @ Sigma12)

    return mu2, Sigma2  # mean and covariance


def GP(X1, y1, X2, kernel_func, sigma_noise=0.0):
    # Kernel of noisy observations
    Sigma11 = kernel_func(X1, X1) + ((sigma_noise ** 2) * np.eye(X1.shape[0]))

    # Kernel of observations vs to-predict
    Sigma12 = kernel_func(X1, X2)

    # Solve. This basically finds (Sigma11^-1 * Sigma12) transposed // assume_a='pos' assumes its positive definite.
    solved = sp.linalg.solve(Sigma11, Sigma12, assume_a='pos').T

    # Compute posterior mean - mu 2|1
    mu2 = solved @ y1
    
    # Compute posterior covariance - Sigma 2|1
    Sigma22 = kernel_func(X2, X2)
    Sigma2 = Sigma22 - (solved @ Sigma12)

    return mu2, Sigma2  # mean and covariance
