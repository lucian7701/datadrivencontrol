"""Taken from https://peterroelants.github.io/posts/gaussian-process-tutorial/"""

from gp_visualisations import sampling_from_a_prior, visualise_sin_posterior
from kernels import ExponentiatedQuadraticKernel


# Visualise the prior
# sampling_from_a_prior()

# Squared exponential with signal variance = 0.5
# sampling_from_a_prior(report_squared_exponential)

# Visualise the posterior

# Adjust parameters as necessary for graphs
test_kernel = ExponentiatedQuadraticKernel(sigma_f=0.05, length_scale=1, sigma_n=0.0)

if __name__ == '__main__':
    visualise_sin_posterior(test_kernel)
