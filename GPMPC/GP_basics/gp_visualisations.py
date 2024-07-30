import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import default_squared_exponential, GP, report_squared_exponential
from plotter import GaussianProcessSinglePlotter


def illustrate_covariance(kernel_func=default_squared_exponential):
    # Illustrate covariance matrix and function

    xlim = (-3, 3)
    X = np.expand_dims(np.linspace(*xlim, 25), 1)  # this leaves X as 2D array shape (25,1)
    Sigma = kernel_func(X, X)

    # Plot covariance matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    im = ax1.imshow(Sigma, cmap=cm.YlGnBu)
    cbar = plt.colorbar(
        im, ax=ax1, fraction=0.045, pad=0.05)
    cbar.ax.set_ylabel('$k(x,x)$', fontsize=10)
    ax1.set_title((
        'Exponentiated quadratic \n'
        'example of covariance matrix'))
    ax1.set_xlabel('x', fontsize=13)
    ax1.set_ylabel('x', fontsize=13)
    ticks = list(range(xlim[0], xlim[1] + 1))
    ax1.set_xticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_yticks(np.linspace(0, len(X) - 1, len(ticks)))
    ax1.set_xticklabels(ticks)
    ax1.set_yticklabels(ticks)
    ax1.grid(False)
    #

    # Show covariance with X=0
    xlim = (-4, 4)
    X = np.expand_dims(np.linspace(*xlim, num=50), 1)
    zero = np.array([[0]])
    Sigma0 = kernel_func(X, zero)

    # Make the plots
    ax2.plot(X[:, 0], Sigma0[:, 0], label='$k(x,0)$')
    ax2.set_xlabel('x', fontsize=13)
    ax2.set_ylabel('covariance', fontsize=13)
    ax2.set_title((
        'Exponentiated quadratic  covariance\n'
        'between $x$ and $0$'))
    ax2.set_ylim([0, 1.1])
    ax2.set_xlim(*xlim)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.show()


def sampling_from_a_prior(kernel_func=default_squared_exponential):
    """
    In practice, we cant sample a full function evaluation f from a GP distribution.
    We can sample function evaluations y of a function f at a finite set of points X.
    Finite subset results in a marginal distribution.
    Sample 5 different realisations from a Gaussian process with exponential quadratic prior
    (without any observed data).
    """

    nb_of_samples = 41
    number_of_functions = 10
    # Independent variable samples
    X = np.expand_dims(np.linspace(-4, 4, nb_of_samples), 1)
    Sigma = kernel_func(X, X)
    # Draw samples
    ys = np.random.multivariate_normal(mean=np.zeros(nb_of_samples), cov=Sigma, size=number_of_functions)

    plotter = GaussianProcessSinglePlotter(
        ys=ys,
        X2=X,
        x_label='X',
        y_label='Y',
        title='10 Sampled Functions',
        xlim=(-4, 4),
        ylim=(-3, 3)
    )
    plotter.plot_sampled_functions()
    plotter.show_plot()


def visualise_sin_posterior(kernel_func=default_squared_exponential):
    f_sin = lambda x: (np.sin(x)).flatten()

    n1 = 8
    n2 = 75
    ny = 10
    domain = (-6, 6)

    X1 = np.random.uniform(domain[0], domain[1], size=(n1, 1))
    y1 = f_sin(X1)

    # Predict points at uniform spacing to capture function
    X2 = np.linspace(-15, 15, n2).reshape(-1, 1)
    # Compute posterior mean and covariance
    mu2, Sigma2 = GP(X1, y1, X2, kernel_func)
    # Compute standard deviation at the test points to be plotted
    sigma2 = np.sqrt(np.diag(Sigma2))
    # Draw some samples of the posterior
    y2 = np.random.multivariate_normal(mean=mu2, cov=Sigma2, size=ny)

    # plot 1 - the posterior
    post_title = 'Distribution of posterior and prior data sin.'
    distribution_plot = GaussianProcessSinglePlotter(
        X1=X1,
        y1=y1,
        X2=X2,
        mu2=mu2,
        sigma2=sigma2,
        actual_func=f_sin,
        ys=y2,
        xlim=(-6,6),
        ylim=(-3, 3),
        title=post_title
    )
    distribution_plot.plot_distribution()
    # distribution_plot.plot_sampled_functions()
    distribution_plot.show_plot()

    # plot 2 - the samples
    sample_title = '10 different function realisations from posterior sin'
    sample_plot = GaussianProcessSinglePlotter(
        ys=y2,
        X2=X2,
        title=sample_title,
        xlim=(-6, 6),
        ylim=(-3, 3)
    )
    sample_plot.plot_sampled_functions()
    sample_plot.show_plot()


if __name__ == '__main__':
    # illustrate_covariance()
    # sampling_from_a_prior()
    visualise_sin_posterior()
    pass
