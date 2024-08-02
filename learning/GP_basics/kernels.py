import numpy as np
import scipy as sp


class Kernel:
    """Base class for kernel functions."""

    def __call__(self, xa, xb):
        raise NotImplementedError("Kernel function must implement __call__ method")

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)


class ExponentiatedQuadraticKernel(Kernel):
    """Exponentiated Quadratic (RBF) kernel function."""

    def __init__(self, sigma_f=1.0, length_scale=1.0, sigma_n=0.0):
        self.sigma_f = sigma_f
        self.length_scale = length_scale
        self.sigma_n = sigma_n

    def __call__(self, xa, xb):
        # Ensure xa and xb have the same dimensionality
        assert xa.shape[1] == xb.shape[1], "Input vectors must have the same dimensionality"

        # Compute the squared Euclidean distance
        sq_norm = sp.spatial.distance.cdist(xa / self.length_scale, xb / self.length_scale, 'sqeuclidean')

        # Compute the covariance matrix
        K = self.sigma_f ** 2 * np.exp(-0.5 * sq_norm)

        # Add noise variance to the diagonal if xa and xb are the same (i.e., during training)
        # This is due to the linearity of expectations of independent random variables.
        # Variance of adding 2 random variables, add individual variances.
        if np.array_equal(xa, xb):
            K += self.sigma_n ** 2 * np.eye(len(xa))

        return K

    def set_params(self, **params):
        super().set_params(**params)


